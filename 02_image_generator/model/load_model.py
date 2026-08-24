from diffusers import StableDiffusionXLPipeline
import torch

MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    pipe.enable_sequential_cpu_offload()

    return pipe, device