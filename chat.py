from talk_ollama import Ollama
from talk_openai import MyOpenAI

if __name__ == "__main__":
    model = MyOpenAI()
    while True:
        user_input = input("You: ")
        if user_input == "exit":
            break
        response = model.llm.invoke(user_input)
        print(f"Ollama: \n {response}")
        