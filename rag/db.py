from __future__ import annotations

import os
import uuid
import hashlib, json
from typing import Any, Optional

import psycopg2
from psycopg2.extras import Json, RealDictCursor


def get_conn():
    return psycopg2.connect(os.environ["DB_URL"])

def make_cache_key(query: str, params: dict) -> str:
    payload = json.dumps({"query": query, **params}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()

def get_cached(cache_key: str) -> dict | None:
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT answer, sources, model
                FROM query_cache
                WHERE cache_key = %s
                  AND (expires_at IS NULL OR expires_at > now())
            """, (cache_key,))
            row = cur.fetchone()
            return dict(row) if row else None
    finally:
        conn.close()

def set_cached(cache_key: str, query: str, answer: str,
               sources: list, model: str, params: dict,
               ttl_hours: int = 24) -> None:
    conn = get_conn()
    try:
        with conn, conn.cursor() as cur:
            cur.execute("""
                INSERT INTO query_cache (cache_key, query, answer, sources, model, params, expires_at)
                VALUES (%s, %s, %s, %s, %s, %s, now() + %s * interval '1 hour')
                ON CONFLICT (cache_key) DO NOTHING
            """, (cache_key, query, answer, Json(sources), model, Json(params), ttl_hours))
    finally:
        conn.close()


def ensure_thread(thread_id: Optional[str], title: str = "New conversation") -> str:
    conn = get_conn()
    try:
        with conn, conn.cursor() as cur:
            if thread_id:
                cur.execute("SELECT id FROM chat_threads WHERE id = %s", (thread_id,))
                if cur.fetchone():
                    cur.execute(
                        "UPDATE chat_threads SET updated_at = now() WHERE id = %s",
                        (thread_id,),
                    )
                    return str(thread_id)

            new_id = str(uuid.uuid4())
            cur.execute(
                """
                INSERT INTO chat_threads (id, title)
                VALUES (%s, %s)
                """,
                (new_id, title),
            )
            return new_id
    finally:
        conn.close()


def update_thread_title(thread_id: str, title: str) -> None:
    conn = get_conn()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE chat_threads
                SET title = %s, updated_at = now()
                WHERE id = %s
                """,
                (title, thread_id),
            )
    finally:
        conn.close()


def insert_message(
    thread_id: str,
    role: str,
    content: str,
    sources: Optional[list[dict]] = None,
) -> str:
    conn = get_conn()
    try:
        with conn, conn.cursor() as cur:
            msg_id = str(uuid.uuid4())
            cur.execute(
                """
                INSERT INTO chat_messages (id, thread_id, role, content, sources)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (msg_id, thread_id, role, content, Json(sources or [])),
            )
            cur.execute(
                "UPDATE chat_threads SET updated_at = now() WHERE id = %s",
                (thread_id,),
            )
            return msg_id
    finally:
        conn.close()


def get_recent_messages(thread_id: str, limit: int = 8) -> list[dict[str, Any]]:
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT role, content, created_at
                FROM chat_messages
                WHERE thread_id = %s
                ORDER BY created_at DESC, id DESC
                LIMIT %s
                """,
                (thread_id, limit),
            )
            rows = cur.fetchall()
            return list(reversed(rows))
    finally:
        conn.close()


def get_threads(limit: int = 50) -> list[dict]:
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, title, created_at, updated_at
                FROM chat_threads
                ORDER BY updated_at DESC
                LIMIT %s
            """, (limit,))
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def get_thread_messages(thread_id: str) -> list[dict]:
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT role, content, sources, created_at
                FROM chat_messages
                WHERE thread_id = %s
                ORDER BY created_at ASC
            """, (thread_id,))
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def insert_rag_run(
    thread_id: str,
    query: str,
    answer: str,
    model: str,
    latency_ms: int,
    used_hybrid: bool,
    used_rerank: bool,
    params: dict,
    sources: list[dict],
) -> None:
    conn = get_conn()
    try:
        with conn, conn.cursor() as cur:
            cur.execute("""
                INSERT INTO rag_runs (
                    thread_id, query_text, answer_text, model_name, latency_ms,
                    hybrid, rerank, top_k_retrieve, top_k_rerank, sources
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                thread_id, query, answer, model, latency_ms,
                used_hybrid, used_rerank,
                params["top_k_retrieve"], params["top_k_rerank"],
                Json(sources),
            ))
    finally:
        conn.close()