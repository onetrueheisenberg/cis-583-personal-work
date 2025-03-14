from openai import OpenAI

client = OpenAI(api_key='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')


def chat_with_gpt(prompt):
    response = client.chat.completions.create(model='gpt-3.5-turbo',
    messages=[{
        "role": 'user',
        'content': prompt
    }])

    return response.choices[0].message.content.strip()

if __name__ == '__main__':
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            break
        response = chat_with_gpt(user_input)
        print("Chatbot:", response)
