from model.load_model import load_model
import torch

# Load the tokenizer, model, and device once
tokenizer, model, device = load_model()


def generate_text(prompt: str) -> str:
    """
    Generate text from a user prompt using the loaded language model.

    Args:
        prompt (str): User input prompt.

    Returns:
        str: Generated text.
    """

    # Format the prompt as a chat conversation (recommended for Qwen Instruct models)
    messages = [
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize the formatted prompt
    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(device)

    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]

    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    )

    return generated_text.strip()
