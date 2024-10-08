import contextlib
import sys
from types import FrameType
from typing import Any, Iterator

import torch
from torch import nn
import torch_tensorrt
from torch_tensorrt import Input


def is_submodule(parent_module: nn.Module, child_module: nn.Module) -> bool:
    """Check if child_module is a submodule of parent_module."""
    return any(module is child_module for module in parent_module.modules())


def get_module_name(model: Any, target_module: nn.Module) -> str | None:
    """Recursively search for the module name within the model."""
    if model is target_module:
        return ""
    if hasattr(model, "__dict__"):
        for name in dir(model):
            attr = getattr(model, name, None)
            if attr is target_module:
                return name
            child_name = get_module_name(attr, target_module)
            if child_name is not None:
                return f"{name}.{child_name}"
    return None


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


class ModuleCompiler:
    def __init__(self, model: Any):
        self.model = model
        self.module_calls: list[tuple[nn.Module, list[torch.Size]]] = []
        self.frames_to_watch: set[int] = set()
        self.frames_to_ignore: set[int] = set()

    def trace_calls(self, frame: FrameType, event: str, arg: Any) -> Any | None:
        del arg
        if event == "call":
            self.handle_call(frame)
        return self.trace_calls

    def handle_call(self, frame: FrameType) -> None:
        code = frame.f_code

        # Ignore built-in functions and methods
        if not code.co_filename:
            return

        # Check if we should ignore this frame
        if id(frame) not in self.frames_to_watch:
            if id(frame.f_back) in (self.frames_to_ignore | self.frames_to_watch):
                self.frames_to_ignore.add(id(frame))
                return

        locals_ = frame.f_locals

        # Check if 'self' is in locals
        self_obj = locals_.get("self")
        if isinstance(self_obj, nn.Module) and code.co_name == "forward":
            # Collect tensor arguments
            args = [
                locals_[var_name]
                for var_name in code.co_varnames
                if var_name != "self" and var_name in locals_
            ]

            # Proceed only if all arguments are tensors
            if all(isinstance(v, torch.Tensor) for v in args):
                # Record the module and input shapes
                shapes = [v.shape for v in args]
                self.module_calls.append((self_obj, shapes))
                self.frames_to_watch.add(id(frame))

    @contextlib.contextmanager
    def enable_tracing(self) -> Iterator[None]:
        sys.settrace(self.trace_calls)
        try:
            yield
        finally:
            sys.settrace(None)

    def run_model(self, func: Any, *args: Any, **kwargs: Any) -> None:
        with self.enable_tracing():
            func(*args, **kwargs)

    def run_model_many(
        self,
        func: Any,
        args_list: list[tuple[Any, ...]],
        kwargs_list: list[dict[str, Any]] | None = None,
    ) -> None:
        if kwargs_list is None:
            kwargs_list = [{}] * len(args_list)
        for args, kwargs in zip(args_list, kwargs_list):
            self.run_model(func, *args, **kwargs)

    def determine_modules_to_compile(self) -> dict[str, list[Input]]:
        module_dict: dict[str, tuple[nn.Module, list[list[torch.Size]]]] = {}
        for module, shapes in self.module_calls:
            module_name = get_module_name(self.model, module)
            if module_name is not None:
                if module_name not in module_dict:
                    module_dict[module_name] = (module, [])
                module_dict[module_name][1].append(shapes)

        # Remove submodules
        modules_to_compile: dict[str, list[Input]] = {}
        for module_name, (module, shape_lists) in module_dict.items():
            if any(
                is_submodule(other_module, module)
                for other_name, (other_module, _) in module_dict.items()
                if other_name != module_name
            ):
                print("warning: is submodule")

            # Transpose shape_lists to get shapes per input
            shapes_per_input = list(zip(*shape_lists))
            trt_inputs = []
            for input_shapes in shapes_per_input:
                unique_shapes = set(input_shapes)
                if len(unique_shapes) == 1:
                    input_shape = unique_shapes.pop()
                    trt_input = Input(shape=input_shape)
                else:
                    dimensions = list(zip(*input_shapes))
                    min_shape = tuple(min(dim) for dim in dimensions)
                    max_shape = tuple(max(dim) for dim in dimensions)
                    opt_shape = max_shape
                    if opt_shape == min_shape:
                        opt_shape = tuple(s + 1 for s in min_shape)
                    trt_input = Input(
                        min_shape=min_shape, opt_shape=opt_shape, max_shape=max_shape
                    )
                trt_inputs.append(trt_input)
            modules_to_compile[module_name] = trt_inputs

        return modules_to_compile

    def compile_and_replace_modules(
        self, modules_to_compile: dict[str, list[Input]]
    ) -> None:
        for module_name, trt_inputs in modules_to_compile.items():
            print(f"Compiling module: {module_name}")
            print(f"TRT Inputs: {trt_inputs}")

            # Get the module from the model
            module = get_module_by_name(self.model, module_name)

            # Compile the module
            compiled_module = torch_tensorrt.compile(
                module, ir="dynamo", inputs=trt_inputs, min_block_size=1
            )

            # Replace the module in the model
            set_module_by_name(self.model, module_name, compiled_module)
            print(f"Replaced module '{module_name}' with its compiled version.\n")
