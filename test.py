from dotenv import load_dotenv
import os
import requests
import json
import google.generativeai as genai
from groq import Groq

load_dotenv()


# GEMINI 2.5 Flash (Google)
print("\nMODEL 1: Gemini 2.5 Flash")
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")
    gemini_response = gemini_model.generate_content("Say hello in one sentence.")
    print("✅ Response:", gemini_response.text)
except Exception as e:
    print("❌ Error:", str(e))


# DeepSeek (OpenRouter)
print("\nMODEL 2: DeepSeek (via OpenRouter)")
try:
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"},
        json={"model": "deepseek/deepseek-chat",
              "messages": [{"role": "user", "content": "Say hello in one sentence."}]}
    )
    print("✅ Response:", r.json()["choices"][0]["message"]["content"])
except Exception as e:
    print("❌ Error:", str(e))

# Llama 3.3 70B (Groq)
print("\nMODEL 3: Llama 3.3 70B (Groq)")
try:
    groq_client = Groq(api_key=os.getenv("LLAMA_API_KEY"))
    groq_response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": "Say hello in one sentence."}
        ]
    )
    print("✅ Response:", groq_response.choices[0].message.content)
except Exception as e:
    print("❌ Error:", str(e))
