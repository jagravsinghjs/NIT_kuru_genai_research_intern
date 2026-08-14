Text Generator — Qwen2.5-7B-Instruct

A simple CLI tool that generates text using Qwen2.5-7B-Instruct, loaded locally through HuggingFace transformers.

app.py runs the main loop — it takes a prompt from the user, passes it to generate_text, and prints the model's reply. Typing "exit" ends the program.

load_model.py loads the tokenizer and model once. The model is loaded in 4-bit NF4 quantization using BitsAndBytesConfig, with device_map="auto" placing it on GPU automatically. This keeps VRAM usage low enough to run on an 8GB card.

generator.py takes the user's prompt, formats it using the model's chat template so it's treated as a proper assistant turn, then generates a response with sampling (temperature 0.7, top_p 0.9), a repetition penalty to avoid repeated text, and a defined stopping token. It decodes only the newly generated tokens and returns them as the output text.