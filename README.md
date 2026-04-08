# Telegram AI Bot

Telegram chat bot powered by OpenRouter and Groq with support for multiple LLMs, vision, persistent memory, and a model switcher.

## Features

- Chat with 10+ free LLMs via OpenRouter (Llama, Qwen, Nemotron, GPT-OSS, and more)
- Vision support via Groq (send images and ask questions)
- Persistent conversation memory with SQLite
- Inline model switcher menu
- Automatic fallback to next model if primary fails
- Personality and tone customization

## Setup

```bash
cp .env.example .env
# fill in your tokens in .env
pip install python-telegram-bot groq requests
python bot.py
```

## Environment Variables

| Variable | Description |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Telegram bot token from @BotFather |
| `OPENROUTER_API_KEY` | OpenRouter API key |
| `GROQ_API_KEY` | Groq API key (used for vision) |
| `HF_API_KEY` | Hugging Face API key (optional) |

## Models

Default model is `stepfun/step-3.5-flash:free`. Switch models at runtime via the inline menu. All listed models are free tier on OpenRouter.
