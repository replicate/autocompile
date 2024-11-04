import torch
from torch import nn
from tracer import ShapeTracer, compile_module


def test_static_shape():
    class StaticModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * 2

    model = StaticModel().eval()
    tracer = ShapeTracer(model)

    # Run the model multiple times with the same input shape
    with tracer.trace():
        model(torch.randn(1, 3))
        model(torch.randn(1, 3))

    modules_to_compile = tracer.determine_modules_to_compile()

    assert len(modules_to_compile) == 1
    assert "" in modules_to_compile  # Root module
    input_spec = modules_to_compile[""][0]
    assert input_spec["shape"] == (1, 3)
    assert input_spec["dtype"] == torch.float32

    print("Static shape test passed.")


def test_dynamic_shape():
    class DynamicModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + 1

    model = DynamicModel().eval()
    tracer = ShapeTracer(model)

    # Run the model multiple times with different input shapes
    with tracer.trace():
        for i in range(1, 5):
            model(torch.randn(i, 3))

    modules_to_compile = tracer.determine_modules_to_compile()

    assert len(modules_to_compile) == 1
    assert "" in modules_to_compile  # Root module
    input_spec = modules_to_compile[""][0]
    assert input_spec["min_shape"] == (1, 3)
    assert input_spec["opt_shape"] == (4, 3)
    assert input_spec["max_shape"] == (4, 3)
    assert input_spec["dtype"] == torch.float32

    print("Dynamic shape test passed.")


def test_complex_pipeline():
    class Tokenizer:
        def __call__(self, texts: list[str]) -> torch.Tensor:
            return torch.tensor([[len(text)] for text in texts], dtype=torch.float32)

    class ImageModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * 2

    # class BadProcessor(nn.Module):
    #     def forward(self, x: torch.Tensor, count: int) -> torch.Tensor:
    #         return torch.stack([x] * count)

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
            #self.image_processor = BadProcessor()
            self.model = UpscalerModel()

        def upscale(self, images: torch.Tensor) -> torch.Tensor:
            #processed_images = self.image_processor(images, 3)
            processed_images = torch.stack([images] * 3)
            return self.model(processed_images )

    class Predictor:
        def __init__(self):
            self.image_pipe = ImagePipeline()
            self.upscaler = UpscalerPipeline()

        def predict(self, prompt: str, n: int) -> torch.Tensor:
            images = self.image_pipe.generate(prompt, n)
            return self.upscaler.upscale(images)

    predictor = Predictor()
    tracer = ShapeTracer(predictor)

    # Run the model multiple times with different inputs
    with tracer.trace():
        predictor.predict("hello", 2)
        predictor.predict("world", 3)

    modules_to_compile = tracer.determine_modules_to_compile()

    expected_modules = {"image_pipe.model", "upscaler.model"}
    assert set(modules_to_compile.keys()) == expected_modules

    # Check the input shapes for 'image_pipe.model'
    image_model_spec = modules_to_compile["image_pipe.model"][0]
    assert image_model_spec["min_shape"] == (2, 1)
    assert image_model_spec["opt_shape"] == (3, 1)
    assert image_model_spec["max_shape"] == (3, 1)
    assert image_model_spec["dtype"] == torch.float32

    # Check the input shapes for 'upscaler.model'
    upscaler_spec = modules_to_compile["upscaler.model"][0]
    assert upscaler_spec["min_shape"] == (3, 2, 1)
    assert upscaler_spec["opt_shape"] == (3, 3, 1)
    assert upscaler_spec["max_shape"] == (3, 3, 1)
    assert upscaler_spec["dtype"] == torch.float32

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
    tracer = ShapeTracer(embedder)

    with tracer.trace():
        for i in range(1, 2):
            embedder.embed(["a dog"] * i)

    modules_to_compile = tracer.determine_modules_to_compile()
    assert modules_to_compile
    print("Modules to compile for CLIP:", modules_to_compile)

    # Optional: test compilation
    compile_module(embedder, modules_to_compile)


if __name__ == "__main__":
    test_static_shape()
    test_dynamic_shape()
    test_complex_pipeline()
    # Uncomment to run CLIP test (requires transformers library and GPU)
    test_clip()
