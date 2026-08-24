from model.load_model import load_model
import torch

pipe, device = load_model()


def generate_image(prompt: str, negative_prompt: str = "", steps: int = 40, guidance_scale: float = 7.5):
    generator = torch.Generator(device=device).manual_seed(torch.seed())

    with torch.inference_mode():
        image = pipe(
            prompt=prompt,
            negative_prompt="blurry, low quality, deformed, disfigured, bad anatomy, extra limbs, watermark, text, ugly",
            num_inference_steps=steps,
            guidance_scale=9,
            generator=generator,
        ).images[0]

    return image