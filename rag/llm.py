"""LLM layer: Groq (free tier) chat completion."""
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
_client = None


def generate(messages: list[dict], temperature: float = 0) -> str:
    global _client
    if _client is None:
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY not set. Add it to the .env file.")
        _client = Groq(api_key=key.strip())
    resp = _client.chat.completions.create(
        model=MODEL, messages=messages, temperature=temperature, max_tokens=2048
    )
    return resp.choices[0].message.content
