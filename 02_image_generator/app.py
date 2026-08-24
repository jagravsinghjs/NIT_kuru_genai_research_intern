from model.generator import generate_image
import os

OUTPUT_DIR = "outputs"


def main():
    print("=" * 50)
    print("Image Generator")
    print("Type 'exit' to quit.")
    print("=" * 50)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    count = 1

    while True:
        prompt = input("\nEnter Prompt: ")

        if prompt.lower() == "exit":
            print("\nGoodbye!")
            break

        print("\nGenerating image...")
        image = generate_image(prompt)

        filename = os.path.join(OUTPUT_DIR, f"image_{count}.png")
        image.save(filename)
        print(f"Saved to {filename}")
        count += 1


if __name__ == "__main__":
    main()