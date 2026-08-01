from model.generator import generate_text


def main():
    print("=" * 50)
    print("Text Generator")
    print("Type 'exit' to quit.")
    print("=" * 50)

    while True:
        prompt = input("\nEnter Prompt: ")

        if prompt.lower() == "exit":
            print("\nGoodbye!")
            break

        response = generate_text(prompt)

        print("\nGenerated Text:\n")
        print(response)


if __name__ == "__main__":
    main()