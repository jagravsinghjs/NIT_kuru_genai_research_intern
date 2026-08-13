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
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
)
    # Convert tokens back to text
    input_length = inputs["input_ids"].shape[-1]

    generated_ids = outputs[0][input_length:]

    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
)
    return generated_text