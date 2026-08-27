from model.generator import generate_caption


def main():
    print("=" * 50)
    print("Image Captioning System")
    print("Type 'exit' to quit.")
    print("=" * 50)

    while True:
        image_path = input("\nEnter image path: ")

        if image_path.lower() == "exit":
            print("\nGoodbye!")
            break

        try:
            caption = generate_caption(image_path)
            print("\nCaption:\n")
            print(caption)
        except FileNotFoundError:
            print("\nFile not found. Check the path and try again.")


if __name__ == "__main__":
    main()