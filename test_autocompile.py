import torch
from torch import nn
from autocompile import ModuleCompiler


def test_static_shape():
    class StaticModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * 2

    model = StaticModel().eval()
    compiler = ModuleCompiler(model)

    # Run the model multiple times with the same input shape
    inputs = [(torch.randn(1, 3),), (torch.randn(1, 3),)]
    compiler.run_model_many(model, inputs)

    modules_to_compile = compiler.determine_modules_to_compile()

    assert len(modules_to_compile) == 1
    trt_inputs = modules_to_compile[""]
    assert len(trt_inputs) == 1
    input_shape = trt_inputs[0].shape
    assert input_shape == torch.Size([1, 3])

    print("Static shape test passed.")


def test_dynamic_shape():
    class DynamicModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + 1

    model = DynamicModel().eval()
    compiler = ModuleCompiler(model)

    # Run the model multiple times with different input shapes
    inputs = [(torch.randn(i, 3),) for i in range(1, 5)]
    compiler.run_model_many(model, inputs)

    modules_to_compile = compiler.determine_modules_to_compile()

    assert len(modules_to_compile) == 1
    trt_inputs = modules_to_compile[""]
    assert len(trt_inputs) == 1
    trt_input = trt_inputs[0]
    expected = {"min_shape": (1, 3), "opt_shape": (4, 3), "max_shape": (4, 3)}
    assert trt_input.shape == expected

    print("Dynamic shape test passed.")


def test_complex_pipeline():
    class Tokenizer:
        def __call__(self, texts: list[str]) -> torch.Tensor:
            return torch.tensor([[len(text)] for text in texts], dtype=torch.float32)

    class ImageModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * 2

    class BadProcessor(nn.Module):
        def forward(self, x: torch.Tensor, count: int) -> torch.Tensor:
            return torch.stack([x] * count)

    class UpscalerModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + 1

    class ImagePipeline:
        def __init__(self):
            self.tokenizer = Tokenizer()
            self.model = ImageModel()

        def generate(self, prompt: str, n: int) -> torch.Tensor:
            tokens = self.tokenizer([prompt] * n)
            return self.model(tokens)

    class UpscalerPipeline:
        def __init__(self):
            self.image_processor = BadProcessor()
            self.model = UpscalerModel()

        def upscale(self, images: torch.Tensor) -> torch.Tensor:
            processed_images = self.image_processor(images, 3)
            return self.model(processed_images)

    class Predictor:
        def __init__(self):
            self.image_pipe = ImagePipeline()
            self.upscaler = UpscalerPipeline()

        def predict(self, prompt: str, n: int) -> torch.Tensor:
            images = self.image_pipe.generate(prompt, n)
            return self.upscaler.upscale(images)

    predictor = Predictor()
    compiler = ModuleCompiler(predictor)

    # Run the model multiple times with different inputs
    args_list = [("hello", 2), ("world", 3)]
    compiler.run_model_many(predictor.predict, args_list)

    modules_to_compile = compiler.determine_modules_to_compile()

    expected_modules = {"image_pipe.model", "upscaler.model"}
    assert set(modules_to_compile.keys()) == expected_modules

    # Check the input shapes for 'image_pipe.model'
    trt_inputs_image = modules_to_compile["image_pipe.model"]
    assert len(trt_inputs_image) == 1
    trt_input_image = trt_inputs_image[0]
    expected = {"min_shape": (2, 1), "opt_shape": (3, 1), "max_shape": (3, 1)}
    assert trt_input_image.shape == expected

    # Check the input shapes for 'upscaler.model'
    trt_inputs_upscaler = modules_to_compile["upscaler.model"]
    assert len(trt_inputs_upscaler) == 1
    trt_input_upscaler = trt_inputs_upscaler[0]
    expected = {"min_shape": (3, 2, 1), "opt_shape": (3, 3, 1), "max_shape": (3, 3, 1)}
    assert trt_input_upscaler.shape == expected
    print("Complex pipeline test passed.")


def test_clip():
    import transformers

    class Embedder:
        def __init__(self):
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            self.clip_model = (
                transformers.CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                .eval()
                .cuda()
            )

        def embed(self, texts: list[str]) -> torch.Tensor:
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to("cuda")
            return self.clip_model.get_text_features(**inputs)

    embedder = Embedder()
    compiler = ModuleCompiler(embedder)
    for i in range(1, 2):
        compiler.run_model(embedder.embed, ["a dog"] * i)
    print(compiler.module_calls)
    modules_to_compile = compiler.determine_modules_to_compile()
    assert modules_to_compile
    print("modules to compile for clip:", modules_to_compile)

    compiler.compile_and_replace_modules(modules_to_compile)


if __name__ == "__main__":
    # Run tests manually
    test_clip()
    # test_static_shape()
    # test_dynamic_shape()
    test_complex_pipeline()
