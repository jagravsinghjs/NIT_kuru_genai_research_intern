from diffusers import StableDiffusionPipeline
import torch

MODEL_NAME = "SG161222/Realistic_Vision_V5.1_noVAE"


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        safety_checker=None,
    )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()

    return pipe, device
