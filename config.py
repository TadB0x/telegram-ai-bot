import os

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
HF_API_KEY = os.getenv("HF_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

ALLOWED_DM_USER_ID = None
MAX_HISTORY_MESSAGES = 40
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Default text model
DEFAULT_MODEL = "stepfun/step-3.5-flash:free"

# Vision via Groq (not OpenRouter)
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Fallback models if primary fails (tried in order)
FALLBACK_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "qwen/qwen3.6-plus:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "openai/gpt-oss-120b:free",
    "z-ai/glm-4.5-air:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "arcee-ai/trinity-large-preview:free",
    "arcee-ai/trinity-mini:free",
    "minimax/minimax-m2.5:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
]

# Competitor bots to track
COMPETITOR_BOTS = ["moreweb"]

# Model menu — all verified available as of 2026-03-30
MODELS = {
    "stepfun/step-3.5-flash:free":                 "⚡ Step 3.5 Flash — default",
    "meta-llama/llama-3.3-70b-instruct:free":      "🦙 Llama 3.3 70B — reliable",
    "qwen/qwen3.6-plus:free":                      "🔮 Qwen3.6 Plus — 1M context",
    "nvidia/nemotron-3-super-120b-a12b:free":      "🧠 Nemotron 120B — very smart",
    "openai/gpt-oss-120b:free":                    "🤖 GPT-OSS 120B",
    "openai/gpt-oss-20b:free":                     "⚡ GPT-OSS 20B — fast",
    "qwen/qwen3-next-80b-a3b-instruct:free":       "🔮 Qwen3 80B",
    "qwen/qwen3-coder:free":                       "💻 Qwen3 Coder — for code",
    "z-ai/glm-4.5-air:free":                       "🌐 GLM-4.5 Air",
    "minimax/minimax-m2.5:free":                   "🌀 MiniMax M2.5",
    "arcee-ai/trinity-large-preview:free":         "🔱 Trinity Large",
    "google/gemma-3-27b-it:free":                  "👁 Gemma 3 27B — vision",
    "nvidia/nemotron-nano-12b-v2-vl:free":         "👁 Nemotron VL — vision",
    "nousresearch/hermes-3-llama-3.1-405b:free":   "🔬 Hermes 405B — max intelligence",
}
