"""
Builds dynamic system prompts from stored group memory.
Analyzes user styles and generates adaptive personality.
"""

import json
import logging
from collections import Counter
import memory

log = logging.getLogger(__name__)

BASE_PROMPT = """You are a member of this private Telegram group — not just a bot, but an actual participant.

Core rules:
- Talk like the people in this group talk. Match their tone, slang, energy, and humor.
- Be casual, direct, and witty. Don't be corporate or formal.
- Keep responses concise unless a detailed answer is needed.
- Use markdown only when it genuinely helps (code blocks, lists).
- You have web search and image generation — use them proactively.
- When someone replies to your message, you're already in a conversation — be natural.
- You read every message in the chat, so you always know what's going on.
- Never say "As an AI" or "I cannot" — just be helpful or honest naturally.
- You can run Python code in a secure isolated Docker sandbox — use it for math, algorithms, data analysis, testing snippets. The sandbox has numpy, pandas, matplotlib but no internet."""


def build_system_prompt(chat_id: int) -> str:
    lines = [BASE_PROMPT]

    # User profiles
    profiles = memory.get_user_profiles(chat_id)
    if profiles:
        lines.append("\n\n## People in this group:\n")
        for p in profiles:
            name = p.get("first_name") or p.get("username") or f"user_{p['user_id']}"
            username = f"@{p['username']}" if p.get("username") else ""
            style = p.get("style_note", "")
            samples = json.loads(p.get("samples") or "[]")[-5:]
            vocab = json.loads(p.get("vocab") or "{}")
            top_words = [w for w, _ in Counter(vocab).most_common(10)]

            entry = f"**{name}** {username} — {p.get('msg_count', 0)} messages"
            if style:
                entry += f"\n  Style: {style}"
            if top_words:
                entry += f"\n  Often uses: {', '.join(top_words)}"
            if samples:
                entry += f"\n  Example messages: {' | '.join(samples[:3])}"
            lines.append(entry)

    # Competitor features to beat
    features = memory.get_competitor_features()
    if features:
        seen = set()
        unique = []
        for f in features:
            key = f["feature"]
            if key not in seen:
                seen.add(key)
                unique.append(f)
        if unique:
            lines.append("\n\n## Other bot features you should match/beat:\n")
            for f in unique[:10]:
                lines.append(f"- [{f['bot_name']}] {f['feature']}: {f.get('example','')[:100]}")

    return "\n".join(lines)


def format_chat_context(messages: list) -> str:
    """Format recent chat messages as readable context."""
    if not messages:
        return ""
    lines = []
    for m in messages[-50:]:
        name = m["first_name"] or m["username"] or "Unknown"
        if m["is_bot"]:
            name = f"[BOT:{name}]"
        text = m["text"] or ""
        media = m["media_desc"]
        if media and not text:
            text = f"[{m['msg_type']}: {media}]"
        elif media and text:
            text = f"[{m['msg_type']}: {media}] {text}"
        if text:
            lines.append(f"{name}: {text}")
    return "\n".join(lines)
