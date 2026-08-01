from model.load_model import load_model
import torch

# Load model once
tokenizer, model, device = load_model()


def generate_text(prompt: str):

    # Convert text into tokens
    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(device)

    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200
        )

    # Convert tokens back to text
    generated_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return generated_text