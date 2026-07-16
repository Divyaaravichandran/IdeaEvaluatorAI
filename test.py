from dotenv import load_dotenv
import os
import requests
from groq import Groq

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")


def call_openrouter(model_name):
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_name,
                "messages": [
                    {"role": "user", "content": "Say hello in one sentence."}
                ],
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ Error: {str(e)}"


def call_groq():
    try:
        client = Groq(api_key=LLAMA_API_KEY)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": "Say hello in one sentence."}
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error: {str(e)}"


# 🚀 Run all models
models = {
    "Qwen3-32B (OpenRouter)": "qwen/qwen3-32b",
    "DeepSeek (OpenRouter)": "deepseek/deepseek-chat",
}

for name, model in models.items():
    print(f"\nMODEL: {name}")
    print("✅ Response:", call_openrouter(model))

print("\nMODEL: Llama 3.3 70B (Groq)")
print("✅ Response:", call_groq())