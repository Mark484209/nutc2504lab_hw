while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "q"]:
        print("Bye!")
        break

    response = client.chat.completions.create(
        model="Qwen/Qwen3-VL-8B-Instruct",
        messages=[
            {"role": "system", "content": "你是一個繁體中文的聊天機器人，請簡潔答覆"},
            {"role": "user", "content": user_input}
        ],
        temperature=0.7,
        max_tokens=256
    )

    print(f"AI : {response.choices[0].message.content}")

