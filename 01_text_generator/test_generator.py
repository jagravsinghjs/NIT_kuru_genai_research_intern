from model.generator import generate_text

prompt = input("Enter a prompt: ")

response = generate_text(prompt)

print("\nGenerated Text:\n")
print(response)
