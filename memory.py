"""
Persistent memory for the bot:
- All chat messages (for context + learning)
- User profiles (name, style, topics, vocabulary)
- Competitor bot feature tracking
"""

import sqlite3
import json
import os
import re
from datetime import datetime
from collections import Counter

DB_PATH = os.path.join(os.path.dirname(__file__), "memory.db")


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS messages (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id    INTEGER NOT NULL,
            user_id    INTEGER,
            username   TEXT,
            first_name TEXT,
            text       TEXT,
            msg_type   TEXT DEFAULT 'text',  -- text/photo/video/sticker/voice
            media_desc TEXT,                 -- AI description of media
            ts         TEXT NOT NULL,
            is_bot     INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS users (
            user_id    INTEGER PRIMARY KEY,
            username   TEXT,
            first_name TEXT,
            last_seen  TEXT,
            msg_count  INTEGER DEFAULT 0,
            -- JSON fields
            topics     TEXT DEFAULT '[]',     -- top topics discussed
            vocab      TEXT DEFAULT '{}',     -- word freq map
            style_note TEXT DEFAULT '',       -- AI-generated style description
            samples    TEXT DEFAULT '[]'      -- recent message samples
        );

        CREATE TABLE IF NOT EXISTS competitor_features (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            bot_name   TEXT,
            feature    TEXT,
            example    TEXT,
            seen_at    TEXT
        );

        CREATE TABLE IF NOT EXISTS bot_notes (
            key   TEXT PRIMARY KEY,
            value TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, ts);
        CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id);
        """)


def log_message(chat_id, user_id, username, first_name, text, msg_type="text", media_desc=None, is_bot=False):
    ts = datetime.utcnow().isoformat()
    with _conn() as c:
        c.execute(
            "INSERT INTO messages (chat_id,user_id,username,first_name,text,msg_type,media_desc,ts,is_bot) VALUES (?,?,?,?,?,?,?,?,?)",
            (chat_id, user_id, username, first_name, text, msg_type, media_desc, ts, int(is_bot))
        )
        if not is_bot and user_id:
            _update_user(c, user_id, username, first_name, text, ts)


def _update_user(conn, user_id, username, first_name, text, ts):
    row = conn.execute("SELECT * FROM users WHERE user_id=?", (user_id,)).fetchone()
    if row:
        vocab = json.loads(row["vocab"] or "{}")
        samples = json.loads(row["samples"] or "[]")
        topics = json.loads(row["topics"] or "[]")
        count = row["msg_count"] + 1
    else:
        vocab, samples, topics, count = {}, [], [], 1

    if text:
        words = re.findall(r'\b\w+\b', text.lower())
        for w in words:
            if len(w) > 2:
                vocab[w] = vocab.get(w, 0) + 1
        # Keep top 100 words
        vocab = dict(Counter(vocab).most_common(100))
        samples.append(text[:200])
        samples = samples[-30:]  # keep last 30

    conn.execute("""
        INSERT INTO users (user_id, username, first_name, last_seen, msg_count, vocab, samples, topics)
        VALUES (?,?,?,?,?,?,?,?)
        ON CONFLICT(user_id) DO UPDATE SET
            username=excluded.username,
            first_name=excluded.first_name,
            last_seen=excluded.last_seen,
            msg_count=excluded.msg_count,
            vocab=excluded.vocab,
            samples=excluded.samples
    """, (user_id, username, first_name, ts, count, json.dumps(vocab), json.dumps(samples), json.dumps(topics)))


def get_recent_messages(chat_id, limit=60):
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM messages WHERE chat_id=? ORDER BY ts DESC LIMIT ?",
            (chat_id, limit)
        ).fetchall()
    return list(reversed(rows))


def get_user_profiles(chat_id):
    """Get profiles of active users in a chat."""
    with _conn() as c:
        user_ids = c.execute(
            "SELECT DISTINCT user_id FROM messages WHERE chat_id=? AND is_bot=0 AND user_id IS NOT NULL",
            (chat_id,)
        ).fetchall()
        profiles = []
        for row in user_ids:
            u = c.execute("SELECT * FROM users WHERE user_id=?", (row["user_id"],)).fetchone()
            if u:
                profiles.append(dict(u))
    return profiles


def log_competitor_feature(bot_name, feature, example):
    ts = datetime.utcnow().isoformat()
    with _conn() as c:
        exists = c.execute(
            "SELECT id FROM competitor_features WHERE bot_name=? AND feature=?",
            (bot_name, feature)
        ).fetchone()
        if not exists:
            c.execute(
                "INSERT INTO competitor_features (bot_name,feature,example,seen_at) VALUES (?,?,?,?)",
                (bot_name, feature, example, ts)
            )


def get_competitor_features(bot_name=None):
    with _conn() as c:
        if bot_name:
            rows = c.execute("SELECT * FROM competitor_features WHERE bot_name=?", (bot_name,)).fetchall()
        else:
            rows = c.execute("SELECT * FROM competitor_features").fetchall()
    return [dict(r) for r in rows]


def update_user_style(user_id, style_note):
    with _conn() as c:
        c.execute("UPDATE users SET style_note=? WHERE user_id=?", (style_note, user_id))


def get_note(key, default=None):
    with _conn() as c:
        row = c.execute("SELECT value FROM bot_notes WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def set_note(key, value):
    with _conn() as c:
        c.execute("INSERT OR REPLACE INTO bot_notes (key,value) VALUES (?,?)", (key, value))


init_db()
