import ollama

MODEL_NAME = "qwen2.5:7b-instruct"


def load_model():
    # Verify that Ollama is available and the model exists.
    try:
        ollama.show(MODEL_NAME)
    except Exception as e:
        raise RuntimeError(
            f"Could not load Ollama model '{MODEL_NAME}'. "
            f"Make sure Ollama is running and the model is installed.\n"
            f"Run: ollama pull {MODEL_NAME}"
        ) from e

    return MODEL_NAME