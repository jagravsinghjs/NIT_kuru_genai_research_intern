from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def load_model():
    """
    Load the tokenizer and language model.

    Returns:
        tokenizer: Hugging Face tokenizer
        model: Loaded language model
        device: CPU or CUDA device
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device
