from model.load_model import load_model
import torch

pipe, device = load_model()


def generate_image(prompt: str, negative_prompt: str = "", steps: int = 30, guidance_scale: float = 7.5):
    generator = torch.Generator(device=device).manual_seed(torch.seed())

    with torch.inference_mode():
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]

    return image