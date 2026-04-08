import requests
from config import GROQ_API_KEY, GROQ_MODEL, SYSTEM_PROMPT


def ask(messages: list[dict], model: str = None) -> str:
    """Send conversation history to Groq and return the reply."""
    payload = {
        "model": model or GROQ_MODEL,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
    }
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]
