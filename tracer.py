import traceback
import contextlib
import inspect
import sys
from types import FrameType
from typing import Any, Generator, TypeAlias
from dataclasses import dataclass
import torch
from torch import nn


@dataclass
class ShapeInfo:
    shape: torch.Size
    dtype: torch.dtype


# Type aliases for better readability
ModuleName: TypeAlias = str
ModuleId: TypeAlias = int
ArgType: TypeAlias = ShapeInfo | type
ArgDict: TypeAlias = dict[str, ArgType]
CallList: TypeAlias = list[ArgDict]
ModuleCalls: TypeAlias = dict[str, CallList]
CompilationSpec: TypeAlias = dict[str, tuple[int, ...] | torch.dtype]


class ShapeTracer:
    def __init__(self, model: Any) -> None:
        self.model = model
        self.module_instance_to_name: dict[ModuleId, ModuleName] = {}
        self.module_input_shapes: dict[ModuleName, ModuleCalls] = {}
        self._build_module_mapping(model, "")

    # actually, it's modules where forward is called only
    # Autoencoder.forward is never called in this code, only autoencoder.{encoder,decoder}.forward
    def _build_module_mapping(
        self, obj: Any, prefix: str, limit: int | None = None
    ) -> None:
        """Recursively build mapping of module instances to their names."""
        print(f"Mapping object: {obj.__class__.__name__} with prefix: {prefix}")
        if isinstance(obj, nn.Module):
            # don't get stuck in the case of cycles
            if id(obj) in self.module_instance_to_name:
                return
            # when we do this we could also grab the signature of forward
            self.module_instance_to_name[id(obj)] = prefix
            print(f"Added module {obj.__class__.__name__} with prefix {prefix}")
            # if this is the first module in a non-module, descend by three levels
            limit = 3 if limit is None else limit
            if limit > 0:
                for name, module in obj.named_children():
                    new_prefix = f"{prefix}.{name}" if prefix else name
                    self._build_module_mapping(module, new_prefix, limit - 1)
        elif hasattr(obj, "__dict__"):
            for name, attr in obj.__dict__.items():
                if isinstance(attr, nn.Module):
                    new_prefix = f"{prefix}.{name}" if prefix else name
                    self._build_module_mapping(attr, new_prefix, 3)
                elif hasattr(attr, "__dict__"):
                    new_prefix = f"{prefix}.{name}" if prefix else name
                    self._build_module_mapping(attr, new_prefix, None)
        print(
            f"Finished mapping object: {obj.__class__.__name__} with prefix: {prefix}"
        )

    @contextlib.contextmanager
    def trace(self) -> Generator[None, None, None]:
        """Context manager to trace the model execution."""

        def trace_func(frame: FrameType, event: str, arg: Any) -> Any:
            if event != "call":
                return trace_func
            self_arg = frame.f_locals.get("self")
            if self_arg is None:
                return trace_func
            if id(self_arg) not in self.module_instance_to_name:
                return trace_func
            func_name = frame.f_code.co_name
            # unfortunately, sometimes top-level modules only use a method called "decode" (etc)
            # in this case we must descend into their components until we get modules where forward is called
            # however, ideally, we would wrap them in an module with mod.forward = vae.decode
            if func_name != "forward":
                return trace_func
            module_name = self.module_instance_to_name[id(self_arg)]
            # .args is args and keyword args in order of declaration
            # .varargs is the name of *args, .keywords is the name of **kwargs
            # .locals has the values but also has local variables
            # we're going to strategically choose to ignore varargs and keywords
            # and also collapse positional and keyword arguments
            arginfo = inspect.getargvalues(frame)
            # get the signature to ignore default values
            params = inspect.signature(getattr(self_arg, func_name)).parameters
            args = {a: arginfo.locals[a] for a in arginfo.args if a != "self"}
            # shape or type for each argument that's not the default
            arg_shapes: ArgDict = {
                k: (
                    ShapeInfo(v.shape, v.dtype)
                    if isinstance(v, torch.Tensor)
                    # else type(v)
                    else v
                )
                for k, v in args.items()
                if v != params[k].default
            }
            # it would have been nice to keep non-tensor arguments for debugging purposes
            self._process_args(module_name, func_name, arg_shapes)
            return trace_func

        sys.settrace(trace_func)
        try:
            yield
        finally:
            sys.settrace(None)

    def _process_args(
        self, module_name: ModuleName, func_name: str, shapes: ArgDict
    ) -> None:
        """Process arguments of a module call and store their shapes."""
        if module_name not in self.module_input_shapes:
            self.module_input_shapes[module_name] = {}
        if func_name not in self.module_input_shapes[module_name]:
            self.module_input_shapes[module_name][func_name] = []
        self.module_input_shapes[module_name][func_name].append(shapes)

    # the reason this works is that we filtered for forward, so all of the calls we have
    # will be for modules where forward is used (e.g. vae.decoder, because only vae.decode is called)
    # later, we'd like to drop the forward filter, and consider wrapping top-level modules where only one method is called
    def determine_modules_to_compile(self) -> dict[ModuleName, list[CompilationSpec]]:
        """Determine which modules to compile and their input specifications."""
        modules_to_compile: dict[ModuleName, list[CompilationSpec]] = {}

        for module_name in sorted(self.module_input_shapes.keys(), key=len):
            if not any(root in module_name for root in modules_to_compile):
                # check if all arguments are tensors, none or bool
                calls: CallList = self.module_input_shapes[module_name]["forward"]
                shape_lists = [
                    [v for v in call.values() if isinstance(v, ShapeInfo)]
                    for call in calls
                ]
                if all(
                    # isinstance(v, ShapeInfo) or v in (type(None), bool)
                    isinstance(v, ShapeInfo) or isinstance(v, ((type(None), bool)))
                    for shapes in shape_lists
                    for v in shapes
                ):
                    input_specs = get_input_specs(shape_lists)
                    modules_to_compile[module_name] = input_specs
        return modules_to_compile


def get_input_specs(shape_lists: list[list[ShapeInfo]]) -> list[CompilationSpec]:
    """Get input specifications for a module."""
    # psych, we actually also want dtype
    input_specs: list[CompilationSpec] = []
    for i in range(len(shape_lists[0])):
        shapes = [shape_list[i] for shape_list in shape_lists]
        if all(shape == shapes[0] for shape in shapes):
            input_specs.append(
                {"shape": tuple(shapes[0].shape), "dtype": shapes[0].dtype}
            )
        else:
            input_specs.append(get_dynamic_range(shapes))
    return input_specs


def get_dynamic_range(shapes: list[ShapeInfo]) -> CompilationSpec:
    min_shape = tuple(
        min(s.shape[d] for s in shapes) for d in range(len(shapes[0].shape))
    )
    max_shape = tuple(
        max(s.shape[d] for s in shapes) for d in range(len(shapes[0].shape))
    )
    opt_shape = tuple(shapes[-1].shape)  # Using the last shape as optimal
    return {
        "min_shape": min_shape,
        "opt_shape": opt_shape,
        "max_shape": max_shape,
        "dtype": shapes[0].dtype,
    }


def get_module_by_name(model: Any, module_name: str) -> nn.Module:
    """Retrieve a module from the model given its dot-separated name."""
    parts = module_name.split(".")
    attr = model
    for part in parts:
        if not hasattr(attr, part):
            raise AttributeError(f"Module '{attr}' has no attribute '{part}'")
        attr = getattr(attr, part)
    return attr


def set_module_by_name(model: Any, module_name: str, new_module: Any) -> None:
    """Set a module in the model given its dot-separated name."""
    parts = module_name.split(".")
    attr = model
    for part in parts[:-1]:
        attr = getattr(attr, part)
    setattr(attr, parts[-1], new_module)


def compile_module(
    model: Any,
    modules_to_compile: dict[ModuleName, list[CompilationSpec]],
    offload=True,
) -> Any:
    import torch_tensorrt

    success = []
    if offload:
        model.to("cpu")

    for module_name, inputs in modules_to_compile.items():
        print(f"Compiling module: {module_name}")
        print(f"TRT Inputs: {inputs}")

        # Get the module from the model
        module = get_module_by_name(model, module_name)
        if offload:
            module.to("cuda")

        # Compile the module
        trt_inputs = [torch_tensorrt.Input(**input) for input in inputs]
        try:
            options = {
                "truncate_long_and_double": True,
                "enabled_precisions": {torch.float32, torch.float16},
            }
            compiled_module = torch_tensorrt.compile(
                module, ir="dynamo", inputs=trt_inputs, options=options
            )
            if offload:
                module.to("cpu")
            # torch_tensorrt.save(...)

            # Replace the module in the model
            set_module_by_name(model, module_name, compiled_module)
            del module
            # gc.collect()
            print(f"Replaced module '{module_name}' with its compiled version.\n")
            success.append(module_name)
        except Exception as e:
            print(traceback.format_exc())

            print(f"Failed to compile '{module_name}': {e}")
    print("successfully compiled", ",".join(success))


# todo:
# save modules
# load by overwriting
# load entire object by pickling
