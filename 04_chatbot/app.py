from model.generator import generate_response


def main():
    print("=" * 50)
    print("Qwen2.5 Chatbot")
    print("Type 'exit' to quit.")
    print("=" * 50)

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() == "exit":
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        try:
            response = generate_response(user_input)

            print("\nAssistant:")
            print(response)

        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()