#!/usr/bin/env python3
"""
Telegram AI Bot — full group awareness, vision, adaptive personality.
"""

import asyncio
import fcntl
import functools
import json
import logging
import os
import subprocess
import sys
import tempfile
import uuid
from collections import defaultdict
from io import BytesIO

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
from telegram.constants import ParseMode, ChatAction
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    CallbackQueryHandler, ContextTypes, filters,
)

import ai_client
import memory
import personality
from config import (
    TELEGRAM_BOT_TOKEN, MAX_HISTORY_MESSAGES,
    ALLOWED_DM_USER_ID, DEFAULT_MODEL, MODELS, COMPETITOR_BOTS,
)

# ── Single-instance lock ──────────────────────────────────────────────────────
def acquire_lock():
    f = open("/tmp/telegram-ai-bot.lock", "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except IOError:
        print("Another instance is running.")
        sys.exit(1)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "bot.log")),
    ],
)
log = logging.getLogger(__name__)

# ── State ─────────────────────────────────────────────────────────────────────
ai_history: dict[tuple, list] = defaultdict(list)
user_model: dict[int, str] = {}
BOT_ID: int = 0        # set on startup
BOT_USERNAME: str = "" # set on startup
followup_store: dict[str, tuple] = {}  # req_id -> (uid, cid, [suggestions])

# ── Helpers ───────────────────────────────────────────────────────────────────
def _is_group(update: Update) -> bool:
    return update.effective_chat.type in ("group", "supergroup")

def _allowed(update: Update) -> bool:
    if update.effective_chat.type == "private":
        if ALLOWED_DM_USER_ID is None:
            return True
        return update.effective_user.id == ALLOWED_DM_USER_ID
    return _is_group(update)

def _is_mentioned(update: Update) -> bool:
    msg = update.message
    if not msg:
        return False
    text = msg.text or msg.caption or ""
    entities = list(msg.entities or []) + list(msg.caption_entities or [])
    for ent in entities:
        if ent.type == "mention":
            mention = text[ent.offset: ent.offset + ent.length].lstrip("@").lower()
            if mention == BOT_USERNAME.lower():
                return True
    return f"@{BOT_USERNAME}".lower() in text.lower()

def _is_reply_to_bot(update: Update) -> bool:
    msg = update.message
    if not msg or not msg.reply_to_message:
        return False
    u = msg.reply_to_message.from_user
    return u is not None and u.id == BOT_ID

def _should_respond(update: Update) -> bool:
    """In groups: respond to mentions or replies to bot. In DMs: always."""
    if not _allowed(update):
        return False
    if _is_group(update):
        return _is_mentioned(update) or _is_reply_to_bot(update)
    return True

def _is_competitor(user) -> bool:
    if not user or not user.is_bot:
        return False
    return any(c in (user.username or "").lower() for c in COMPETITOR_BOTS)

def _model_keyboard(uid: int) -> InlineKeyboardMarkup:
    current = user_model.get(uid, DEFAULT_MODEL)
    rows = []
    for mid, label in MODELS.items():
        tick = "✅ " if mid == current else ""
        rows.append([InlineKeyboardButton(f"{tick}{label}", callback_data=f"model:{mid}")])
    return InlineKeyboardMarkup(rows)

async def _download_photo(bot, file_id: str) -> bytes | None:
    try:
        f = await bot.get_file(file_id)
        buf = BytesIO()
        await f.download_to_memory(buf)
        return buf.getvalue()
    except Exception as e:
        log.warning(f"Photo download: {e}")
        return None

async def _video_frame(bot, file_id: str) -> bytes | None:
    try:
        f = await bot.get_file(file_id)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            await f.download_to_drive(tmp.name)
            path = tmp.name
        out = path + ".jpg"
        r = subprocess.run(
            ["ffmpeg", "-i", path, "-frames:v", "1", "-q:v", "3", out, "-y"],
            capture_output=True, timeout=30,
        )
        os.unlink(path)
        if r.returncode == 0 and os.path.exists(out):
            data = open(out, "rb").read()
            os.unlink(out)
            return data
    except Exception as e:
        log.warning(f"Video frame: {e}")
    return None

# ── Observer — runs on ALL messages ──────────────────────────────────────────
async def observe(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg or not msg.from_user:
        return

    user = msg.from_user
    cid  = msg.chat_id
    text = msg.text or msg.caption or ""
    msg_type = "text"
    media_desc = None

    if msg.photo:
        msg_type = "photo"
        photo_bytes = await _download_photo(ctx.bot, msg.photo[-1].file_id)
        if photo_bytes:
            loop = asyncio.get_event_loop()
            try:
                media_desc = await loop.run_in_executor(None, ai_client.describe_image, photo_bytes)
            except Exception:
                media_desc = "photo"
    elif msg.video or msg.video_note:
        msg_type = "video"
        vid = msg.video or msg.video_note
        frame = await _video_frame(ctx.bot, vid.file_id)
        if frame:
            loop = asyncio.get_event_loop()
            try:
                media_desc = await loop.run_in_executor(None, ai_client.describe_image, frame)
            except Exception:
                media_desc = "video"
        else:
            media_desc = "video"
    elif msg.sticker:
        msg_type = "sticker"
        media_desc = msg.sticker.emoji or "sticker"
    elif msg.voice or msg.audio:
        msg_type = "voice"
        voice_file = msg.voice or msg.audio
        try:
            vf = await ctx.bot.get_file(voice_file.file_id)
            vbuf = BytesIO()
            await vf.download_to_memory(vbuf)
            loop = asyncio.get_event_loop()
            transcript = await loop.run_in_executor(
                None, functools.partial(ai_client.transcribe_audio, vbuf.getvalue(), "voice.ogg")
            )
            media_desc = f"voice: \"{transcript}\"" if transcript else "voice message"
        except Exception:
            media_desc = "voice message"
    elif msg.document:
        msg_type = "document"
        media_desc = msg.document.file_name or "file"

    memory.log_message(cid, user.id, user.username, user.first_name,
                       text, msg_type, media_desc, user.is_bot)

    # Competitor tracking
    if user.is_bot and _is_competitor(user) and text:
        bot_name = user.username or user.first_name or "bot"
        if text.startswith("/"):
            cmd = text.split()[0].lstrip("/").split("@")[0]
            if cmd:
                memory.log_competitor_feature(bot_name, f"command /{cmd}", text[:200])
        for kw, feat in [
            ("image","image generation"),("search","web search"),("weather","weather"),
            ("translate","translation"),("summarize","summarization"),("code","code execution"),
            ("audio","audio/voice"),("video","video processing"),("reminder","reminders"),
            ("poll","polls"),("chart","charts"),("news","news feed"),
        ]:
            if kw in text.lower():
                memory.log_competitor_feature(bot_name, feat, text[:200])


# ── Core AI responder ─────────────────────────────────────────────────────────
async def _keep_typing(bot, chat_id: int, stop_event: asyncio.Event):
    """Send typing action every 4s until stop_event is set."""
    while not stop_event.is_set():
        try:
            await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        except Exception:
            pass
        await asyncio.sleep(4)

async def _respond(update: Update, ctx: ContextTypes.DEFAULT_TYPE, image_bytes=None):
    msg = update.message
    if not msg or not msg.from_user:
        return

    uid = msg.from_user.id
    cid = msg.chat_id
    key = (cid, uid)

    text = msg.text or msg.caption or ""
    text = text.replace(f"@{BOT_USERNAME}", "").strip()
    if not text and not image_bytes:
        text = "[sent media]"

    ai_history[key].append({"role": "user", "content": text})
    if len(ai_history[key]) > MAX_HISTORY_MESSAGES:
        ai_history[key] = ai_history[key][-MAX_HISTORY_MESSAGES:]

    # Keep "typing..." alive for the full duration of the request
    stop_typing = asyncio.Event()
    typing_task = asyncio.create_task(_keep_typing(ctx.bot, cid, stop_typing))

    recent = memory.get_recent_messages(cid, limit=60)
    ctx_str = personality.format_chat_context(recent)
    system  = personality.build_system_prompt(cid)
    if ctx_str:
        system += f"\n\n## Recent chat:\n{ctx_str}"

    model = user_model.get(uid, DEFAULT_MODEL)

    try:
        loop = asyncio.get_event_loop()
        reply, new_model, gen_image = await loop.run_in_executor(
            None,
            functools.partial(ai_client.ask, ai_history[key],
                              model=model, system_prompt=system, image_bytes=image_bytes),
        )
    except Exception as e:
        log.exception(f"ask() failed: {e}")
        if ai_history[key] and ai_history[key][-1]["role"] == "user":
            ai_history[key].pop()
        stop_typing.set()
        typing_task.cancel()
        await msg.reply_text("⚠️ AI error. Try again or use /setmodel to switch.")
        return
    finally:
        stop_typing.set()
        typing_task.cancel()

    if new_model != model:
        user_model[uid] = new_model
        await msg.reply_text(f"🔄 Switched to `{MODELS.get(new_model, new_model)}`",
                             parse_mode=ParseMode.MARKDOWN)

    ai_history[key].append({"role": "assistant", "content": reply or ""})

    # Generate follow-up suggestions (non-blocking, best-effort)
    followup_kb = None
    if reply and not gen_image:
        try:
            ctx_snippet = personality.format_chat_context(memory.get_recent_messages(cid, limit=10))
            suggestions = await asyncio.get_event_loop().run_in_executor(
                None,
                functools.partial(ai_client.get_followups, text, reply, ctx_snippet),
            )
            if suggestions:
                req_id = uuid.uuid4().hex[:10]
                followup_store[req_id] = (uid, cid, suggestions)
                # Prune old entries (keep last 200)
                if len(followup_store) > 200:
                    oldest = list(followup_store.keys())[:len(followup_store) - 200]
                    for k in oldest:
                        followup_store.pop(k, None)
                rows = [[InlineKeyboardButton(s, callback_data=f"fq:{req_id}:{i}")]
                        for i, s in enumerate(suggestions)]
                followup_kb = InlineKeyboardMarkup(rows)
        except Exception as e:
            log.warning(f"followup suggestions failed: {e}")

    if gen_image:
        bio = BytesIO(gen_image)
        bio.name = "image.jpg"
        await msg.reply_photo(photo=bio, caption=(reply or "")[:1024])
    elif reply:
        chunks = [reply[i:i+4000] for i in range(0, len(reply), 4000)]
        for i, chunk in enumerate(chunks):
            kb = followup_kb if i == len(chunks) - 1 else None
            try:
                await msg.reply_text(chunk, parse_mode=ParseMode.MARKDOWN, reply_markup=kb)
            except Exception:
                await msg.reply_text(chunk, reply_markup=kb)
    else:
        await msg.reply_text("🤔 No response. Try again.")


# ── Extract image from replied-to message (if any) ───────────────────────────
async def _image_from_replied(update: Update, ctx) -> bytes | None:
    """If the user is replying to a message that contains a photo/video, grab it."""
    replied = update.message.reply_to_message if update.message else None
    if not replied:
        return None
    if replied.photo:
        return await _download_photo(ctx.bot, replied.photo[-1].file_id)
    if replied.video or replied.video_note:
        vid = replied.video or replied.video_note
        return await _video_frame(ctx.bot, vid.file_id)
    return None


# ── Message handlers ──────────────────────────────────────────────────────────
async def handle_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _should_respond(update):
        return
    # If replying to a photo/video message, analyse that media
    image_bytes = await _image_from_replied(update, ctx)
    await _respond(update, ctx, image_bytes=image_bytes)

async def handle_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _should_respond(update):
        return
    photo = await _download_photo(ctx.bot, update.message.photo[-1].file_id)
    await _respond(update, ctx, image_bytes=photo)

async def handle_video(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _should_respond(update):
        return
    vid = update.message.video or update.message.video_note
    if not vid:
        return
    frame = await _video_frame(ctx.bot, vid.file_id)
    await _respond(update, ctx, image_bytes=frame)

async def handle_voice(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _should_respond(update):
        return
    msg = update.message
    voice = msg.voice or msg.audio
    if not voice:
        return
    try:
        f = await ctx.bot.get_file(voice.file_id)
        buf = BytesIO()
        await f.download_to_memory(buf)
        audio_bytes = buf.getvalue()
    except Exception as e:
        log.warning(f"Voice download: {e}")
        await msg.reply_text("⚠️ Couldn't download the voice message.")
        return

    loop = asyncio.get_event_loop()
    transcript = await loop.run_in_executor(
        None, functools.partial(ai_client.transcribe_audio, audio_bytes, "voice.ogg")
    )
    if not transcript:
        await msg.reply_text("⚠️ Couldn't transcribe the voice message.")
        return

    # Show what was heard, then respond
    await msg.reply_text(f"🎙 *Heard:* _{transcript}_", parse_mode=ParseMode.MARKDOWN)
    # Inject transcript as the message text so _respond uses it
    msg.text = transcript
    await _respond(update, ctx)


# ── Commands — NO mention required in groups ──────────────────────────────────
async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _allowed(update):
        return
    uid = update.effective_user.id
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("🤖 Set Model", callback_data="show_models"),
         InlineKeyboardButton("🗑 Clear", callback_data="clear_confirm")],
        [InlineKeyboardButton("ℹ️ Help", callback_data="show_help"),
         InlineKeyboardButton("📊 Status", callback_data="show_status")],
    ])
    await update.message.reply_text(
        "👋 *What's good.* I read this whole chat and I'll reply when you mention me or reply to me.\n\n"
        f"Model: `{user_model.get(uid, DEFAULT_MODEL)}`",
        parse_mode=ParseMode.MARKDOWN, reply_markup=kb,
    )

async def cmd_setmodel(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _allowed(update):
        return
    uid = update.effective_user.id
    await update.message.reply_text(
        f"🤖 *Pick a model* — current: `{user_model.get(uid, DEFAULT_MODEL)}`",
        parse_mode=ParseMode.MARKDOWN, reply_markup=_model_keyboard(uid),
    )

async def cmd_clear(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _allowed(update):
        return
    kb = InlineKeyboardMarkup([[
        InlineKeyboardButton("✅ Clear", callback_data="clear_do"),
        InlineKeyboardButton("❌ Cancel", callback_data="clear_cancel"),
    ]])
    await update.message.reply_text("Clear your conversation history?", reply_markup=kb)

async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _allowed(update):
        return
    uid = update.effective_user.id
    cid = update.effective_chat.id
    model = user_model.get(uid, DEFAULT_MODEL)
    hist  = len(ai_history.get((cid, uid), []))
    total = len(memory.get_recent_messages(cid, limit=9999))
    profs = len(memory.get_user_profiles(cid))
    feats = len(set(f["feature"] for f in memory.get_competitor_features()))
    await update.message.reply_text(
        f"📊 *Status*\n\nModel: `{model}`\nYour history: {hist} msgs\n"
        f"Chat logged: {total} msgs\nUsers profiled: {profs}\nCompetitor features: {feats}",
        parse_mode=ParseMode.MARKDOWN,
    )

async def cmd_users(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _allowed(update):
        return
    profiles = memory.get_user_profiles(update.effective_chat.id)
    if not profiles:
        await update.message.reply_text("No users profiled yet.")
        return
    lines = ["👥 *Group members:*\n"]
    for p in profiles:
        name  = p.get("first_name") or p.get("username") or f"user_{p['user_id']}"
        uname = f" (@{p['username']})" if p.get("username") else ""
        style = p.get("style_note", "").strip()
        lines.append(f"• *{name}*{uname} — {p.get('msg_count',0)} msgs"
                     + (f"\n  _{style}_" if style else ""))
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)

async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _allowed(update):
        return
    await update.message.reply_text(
        "📖 *Commands*\n\n"
        "/start — welcome\n"
        "/setmodel — pick AI model\n"
        "/clear — clear history\n"
        "/status — stats\n"
        "/users — group member profiles\n"
        "/help — this\n\n"
        "💡 Mention me or reply to me to chat.\n"
        "Send photos/videos for vision analysis.\n"
        "Say *draw X* to generate an image.\n"
        "Say *run this code* to execute Python.",
        parse_mode=ParseMode.MARKDOWN,
    )


# ── Callbacks ─────────────────────────────────────────────────────────────────
async def on_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    uid = q.from_user.id
    cid = q.message.chat_id
    data = q.data

    if data.startswith("model:"):
        mid = data[6:]
        if mid in MODELS:
            user_model[uid] = mid
            await q.edit_message_text(f"✅ Model set to `{MODELS[mid]}`",
                                      parse_mode=ParseMode.MARKDOWN)
        else:
            await q.edit_message_text("❌ Unknown model.")

    elif data == "show_models":
        await q.edit_message_text(
            f"🤖 *Pick a model*\nCurrent: `{user_model.get(uid, DEFAULT_MODEL)}`",
            parse_mode=ParseMode.MARKDOWN, reply_markup=_model_keyboard(uid),
        )

    elif data == "show_status":
        hist  = len(ai_history.get((cid, uid), []))
        profs = len(memory.get_user_profiles(cid))
        await q.edit_message_text(
            f"Model: `{user_model.get(uid, DEFAULT_MODEL)}`\n"
            f"History: {hist} msgs | Users: {profs}",
            parse_mode=ParseMode.MARKDOWN,
        )

    elif data == "show_help":
        await q.edit_message_text(
            "Mention me or reply to me.\nSend photos/videos for analysis.\n"
            "Say 'draw X' to generate images.\n/setmodel to switch models."
        )

    elif data == "clear_confirm":
        kb = InlineKeyboardMarkup([[
            InlineKeyboardButton("✅ Yes", callback_data="clear_do"),
            InlineKeyboardButton("❌ No", callback_data="clear_cancel"),
        ]])
        await q.edit_message_text("Clear history?", reply_markup=kb)

    elif data == "clear_do":
        ai_history.pop((cid, uid), None)
        await q.edit_message_text("✅ Done.")

    elif data == "clear_cancel":
        await q.edit_message_text("Cancelled.")

    elif data.startswith("fq:"):
        # Follow-up button tapped — treat it as a new user message
        parts = data.split(":", 2)
        if len(parts) == 3:
            req_id, idx_str = parts[1], parts[2]
            stored = followup_store.get(req_id)
            if stored:
                _, _, suggestions = stored
                try:
                    suggestion = suggestions[int(idx_str)]
                except (IndexError, ValueError):
                    await q.answer("Expired.")
                    return
                # Edit the buttons away (keep the answer text)
                try:
                    await q.edit_message_reply_markup(reply_markup=None)
                except Exception:
                    pass
                # Inject follow-up as a fake user message and respond
                key = (cid, uid)
                ai_history[key].append({"role": "user", "content": suggestion})
                if len(ai_history[key]) > MAX_HISTORY_MESSAGES:
                    ai_history[key] = ai_history[key][-MAX_HISTORY_MESSAGES:]
                # Send the chosen text as a visible message, then respond
                sent = await q.message.reply_text(f"👤 {q.from_user.first_name}: {suggestion}")
                # Build a fake Update-like context for _respond — easier to just call AI directly
                model = user_model.get(uid, DEFAULT_MODEL)
                recent = memory.get_recent_messages(cid, limit=60)
                ctx_str = personality.format_chat_context(recent)
                system = personality.build_system_prompt(cid)
                if ctx_str:
                    system += f"\n\n## Recent chat:\n{ctx_str}"
                stop_typing = asyncio.Event()
                typing_task = asyncio.create_task(_keep_typing(ctx.bot, cid, stop_typing))
                try:
                    loop = asyncio.get_event_loop()
                    reply_text, new_model, gen_image = await loop.run_in_executor(
                        None,
                        functools.partial(ai_client.ask, ai_history[key],
                                          model=model, system_prompt=system),
                    )
                finally:
                    stop_typing.set()
                    typing_task.cancel()
                if new_model != model:
                    user_model[uid] = new_model
                ai_history[key].append({"role": "assistant", "content": reply_text or ""})
                # Follow-up buttons for the new reply
                followup_kb2 = None
                if reply_text:
                    try:
                        ctx_snippet = personality.format_chat_context(memory.get_recent_messages(cid, limit=10))
                        suggestions2 = await loop.run_in_executor(
                            None, functools.partial(ai_client.get_followups, suggestion, reply_text, ctx_snippet)
                        )
                        if suggestions2:
                            req_id2 = uuid.uuid4().hex[:10]
                            followup_store[req_id2] = (uid, cid, suggestions2)
                            followup_kb2 = InlineKeyboardMarkup([
                                [InlineKeyboardButton(s, callback_data=f"fq:{req_id2}:{i}")]
                                for i, s in enumerate(suggestions2)
                            ])
                    except Exception:
                        pass
                if gen_image:
                    bio = BytesIO(gen_image)
                    bio.name = "image.jpg"
                    await sent.reply_photo(photo=bio, caption=(reply_text or "")[:1024])
                elif reply_text:
                    try:
                        await sent.reply_text(reply_text, parse_mode=ParseMode.MARKDOWN, reply_markup=followup_kb2)
                    except Exception:
                        await sent.reply_text(reply_text, reply_markup=followup_kb2)
                else:
                    await sent.reply_text("🤔 No response.")
            else:
                await q.answer("Expired — ask again.")


# ── Background: style update every 30 min ────────────────────────────────────
async def _style_loop():
    try:
      await asyncio.sleep(120)
    except asyncio.CancelledError:
        return
    while True:
        try:
            import sqlite3
            with sqlite3.connect(memory.DB_PATH) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute("SELECT user_id, first_name, username, samples FROM users").fetchall()
            loop = asyncio.get_event_loop()
            for row in rows:
                try:
                    samples = json.loads(row["samples"] or "[]")
                except Exception:
                    samples = []
                if len(samples) < 5:
                    continue
                name = row["first_name"] or row["username"] or f"user_{row['user_id']}"
                sample_text = "\n".join(f'- "{s}"' for s in samples[-20:])
                prompt = (f"Describe '{name}'s texting style in one sentence (max 20 words) "
                          f"based on:\n{sample_text}\nNo preamble.")
                try:
                    style, _, _ = await loop.run_in_executor(
                        None, functools.partial(ai_client.ask,
                                               [{"role": "user", "content": prompt}])
                    )
                    if style:
                        memory.update_user_style(row["user_id"], style.strip())
                        log.info(f"[style] {name}: {style[:60]}")
                except Exception as e:
                    log.warning(f"[style] {name}: {e}")
        except Exception as e:
            log.warning(f"[style loop] {e}")
        try:
            await asyncio.sleep(1800)
        except asyncio.CancelledError:
            return


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    acquire_lock()

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    async def post_init(application):
        global BOT_ID, BOT_USERNAME
        me = await application.bot.get_me()
        BOT_ID = me.id
        BOT_USERNAME = me.username
        log.info(f"Bot: @{BOT_USERNAME} (id={BOT_ID})")

        await application.bot.set_my_commands([
            BotCommand("start",    "Welcome"),
            BotCommand("setmodel", "Switch AI model"),
            BotCommand("clear",    "Clear your history"),
            BotCommand("status",   "Stats"),
            BotCommand("users",    "Group member profiles"),
            BotCommand("help",     "Help"),
        ])
        application.bot_data["style_task"] = asyncio.create_task(_style_loop())

    async def post_shutdown(application):
        task = application.bot_data.get("style_task")
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    app.post_init = post_init
    app.post_shutdown = post_shutdown

    # Group 0: observe everything (logging + competitor tracking)
    app.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND, observe), group=0)
    app.add_handler(MessageHandler(filters.COMMAND, observe), group=0)

    # Commands — no mention needed
    for cmd, fn in [
        ("start", cmd_start), ("help", cmd_help), ("setmodel", cmd_setmodel),
        ("model", cmd_setmodel), ("clear", cmd_clear), ("status", cmd_status),
        ("users", cmd_users),
    ]:
        app.add_handler(CommandHandler(cmd, fn), group=1)

    # AI handlers — group 2
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text), group=2)
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo), group=2)
    app.add_handler(MessageHandler(filters.VIDEO | filters.VIDEO_NOTE, handle_video), group=2)
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice), group=2)
    app.add_handler(CallbackQueryHandler(on_callback), group=1)

    log.info("Starting...")
    app.run_polling(drop_pending_updates=False, allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
