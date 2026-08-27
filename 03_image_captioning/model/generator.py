from model.load_model import load_model
from PIL import Image
import torch

processor, model, device = load_model()


def generate_caption(image_path: str):
    image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=50,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3,
        )

    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    return caption