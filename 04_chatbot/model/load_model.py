import ollama

MODEL_NAME = "qwen2.5:7b-instruct"


def load_model():
    try:
        ollama.show(MODEL_NAME)
    except Exception as e:
        raise RuntimeError(
            f"Could not load Ollama model '{MODEL_NAME}'.\n"
            f"Make sure Ollama is running and the model is installed.\n"
            f"Run: ollama pull {MODEL_NAME}"
        ) from e

    return MODEL_NAME