from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

MODEL_NAME = "Salesforce/blip-image-captioning-base"


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    return processor, model, device
if __name__ == "__main__":
    processor, model, device = load_model()
    print("Model loaded successfully on", device)