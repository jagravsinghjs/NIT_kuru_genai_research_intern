from model.load_model import load_model
import ollama


MODEL_NAME = load_model()


def generate_response(messages: list):
    """
    Generate a response from Qwen2.5 using the conversation history.

    Args:
        messages: List of dictionaries in Ollama chat format.

    Returns:
        Generated assistant response as a string.
    """

    response = ollama.chat(
        model=MODEL_NAME,
        messages=messages,
    )

    return response["message"]["content"]