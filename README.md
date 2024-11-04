# autocompile
Use tracing to automatically infer static and dynamic shapes as well as which modules need to be compiled in nested pipelies objects, then compile with Torch-TensorRT.

Example usage:

```python
In [1]: from diffusers import StableDiffusionPipeline
   ...: import tracer

In [2]: pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
Loading pipeline components...: 100%|████████████████████████████████████████████████████| 7/7 [00:00<00:00, 15.35it/s]

In [3]: pipe_tracer = tracer.ShapeTracer(pipe)
Mapping object: StableDiffusionPipeline with prefix:
Mapping object: AutoencoderKL with prefix: vae
Added module AutoencoderKL with prefix vae
Mapping object: Encoder with prefix: vae.encoder
Added module Encoder with prefix vae.encoder
...
Finished mapping object: VaeImageProcessor with prefix: image_processor
Finished mapping object: StableDiffusionPipeline with prefix:

In [4]: with pipe_tracer.trace():
   ...:     pipe("astronaut riding a horse", num_inference_steps=1)
   ...:
100%|████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:04<00:00,  4.26s/it]

In [5]: pipe_tracer.determine_modules_to_compile()
Out[5]:
{'unet': [{'shape': (2, 4, 64, 64), 'dtype': torch.float32},
  {'shape': (), 'dtype': torch.int64},
  {'shape': (2, 77, 768), 'dtype': torch.float32}],
 'vae.decoder': [{'shape': (1, 4, 64, 64), 'dtype': torch.float32}],
 'text_encoder': [{'shape': (1, 77), 'dtype': torch.int64}],
 'safety_checker': [{'shape': (1, 3, 224, 224), 'dtype': torch.float32},
  {'shape': (1, 3, 512, 512), 'dtype': torch.float32}],
 'vae.post_quant_conv': [{'shape': (1, 4, 64, 64), 'dtype': torch.float32}]}

In [6]: tracer.compile_module(pipe, pipe_tracer.determine_modules_to_compile()) # not implemented yet
```

<details><summary>Outputs for flux</summary>

```python
{
    "t5": [],
    "clip": [],
    "flux": [
        {
            "min_shape": (1, 3808, 64),
            "max_shape": (1, 4096, 64),
            "dtype": torch.bfloat16,
        },
        {
            "min_shape": (1, 3808, 3),
            "max_shape": (1, 4096, 3),
            "dtype": torch.float32,
        },
        {"shape": (1, 256, 4096), "dtype": torch.bfloat16},
        {"shape": (1, 256, 3), "dtype": torch.float32},
        {"shape": (1,), "dtype": torch.bfloat16},
        {"shape": (1, 768), "dtype": torch.bfloat16},
        {"shape": (1,), "dtype": torch.bfloat16},
    ],
    "ae.decoder": [
        {
            "min_shape": (1, 16, 112, 112),
            "max_shape": (1, 16, 136, 136),
            "dtype": torch.float32,
        }
    ],
}
```

```python
{
    "t5": {"forward": [{"text": ["a cool dog"]}, {"text": ["a cool dog"]}]},
    "t5.hf_module": {
        "forward": [
            {
                "input_ids": ShapeInfo(shape=torch.Size([1, 256]), dtype=torch.int64),
                "output_hidden_states": False,
            },
            {
                "input_ids": ShapeInfo(shape=torch.Size([1, 256]), dtype=torch.int64),
                "output_hidden_states": False,
            },
        ]
    },
    ...
}
```
</details>
