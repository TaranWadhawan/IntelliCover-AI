import httpx
import os
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("HUGGINGFACE_TOKEN")

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
}

models = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "HuggingFaceH4/zephyr-7b-beta",
    "microsoft/Phi-3-mini-4k-instruct",
    "Qwen/Qwen2.5-7B-Instruct"
]

for m in models:
    print(f"Testing {m}...")
    try:
        resp = httpx.post(
            "https://router.huggingface.co/v1/chat/completions",
            headers=headers,
            json={
                "model": m,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10
            },
            timeout=10
        )
        print("  Status:", resp.status_code)
        print("  Response:", resp.text)
    except Exception as e:
        print("  Error:", e)
