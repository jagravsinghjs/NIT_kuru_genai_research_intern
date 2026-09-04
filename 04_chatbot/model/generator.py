import ollama
from model.load_model import load_model


MODEL_NAME = load_model()


def generate_response(user_input: str):
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": user_input,
            }
        ],
    )

    return response["message"]["content"]