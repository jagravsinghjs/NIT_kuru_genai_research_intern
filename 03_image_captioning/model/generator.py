from model.load_model import load_model
from PIL import Image
import torch

processor, model, device = load_model()


def generate_caption(image_path: str):
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((1024, 1024))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image in detail."},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=200)

    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    caption = processor.decode(generated_ids[0], skip_special_tokens=True)

    return caption