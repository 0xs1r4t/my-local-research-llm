from __future__ import annotations

import os
import time
import traceback
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from llm.inference import generate
from rag.db import (
    ensure_thread,
    get_recent_messages,
    get_thread_messages,
    get_threads,
    insert_message,
    insert_rag_run,
    update_thread_title,
    set_cached,
    get_cached,
    make_cache_key,
)
from rag.prompts import build_prompt
from retrieval.hybrid import search
from retrieval.rerank import rerank

APP_TITLE = "Dissertation RAG API"
APP_VERSION = "0.1.0"
MODEL_NAME = os.getenv("MODEL_NAME", "llama.cpp-local")


app = FastAPI(title=APP_TITLE, version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    thread_id: Optional[str] = None
    query: str = Field(..., min_length=1)
    hybrid: bool = True
    rerank: bool = False
    stream: bool = False
    top_k_retrieve: int = 10
    top_k_rerank: int = 5
    history_limit: int = 8


class SourceMetadata(BaseModel):
    raw_source: Optional[str] = None
    title: Optional[str] = None
    page: Optional[int] = None
    chunk_id: Optional[str] = None


class SourceItem(BaseModel):
    source: str
    score: Optional[float] = None
    text: Optional[str] = None
    metadata: Optional[SourceMetadata] = None


class QueryResponse(BaseModel):
    thread_id: str
    answer: str
    sources: List[SourceItem]
    model: str
    used_hybrid: bool
    used_rerank: bool


def normalize_source(item: Any) -> SourceItem:
    if isinstance(item, dict):
        nested_meta = item.get("metadata") or {}
        raw_source = (
            item.get("raw_source")
            or item.get("source")
            or nested_meta.get("raw_source")
        )
        title = item.get("title") or nested_meta.get("title")
        return SourceItem(
            source=title or raw_source or "unknown",
            score=item.get("score"),
            text=item.get("text"),
            metadata=SourceMetadata(
                raw_source=raw_source,
                title=title,
                page=item.get("page") or nested_meta.get("page"),
                chunk_id=item.get("chunk_id") or nested_meta.get("chunk_id"),
            ),
        )
    fallback = str(item)
    return SourceItem(
        source=fallback,
        metadata=SourceMetadata(raw_source=fallback, title=fallback),
    )


def ask(
    query: str,
    top_k_retrieve: int = 10,
    top_k_rerank: int = 5,
    use_hybrid: bool = True,
    use_rerank: bool = True,
    history: Optional[List[dict]] = None,
) -> Dict[str, Any]:
    results = search(query, top_k=top_k_retrieve, hybrid=use_hybrid)
    ranked = rerank(query, results, top_k=top_k_rerank) if use_rerank and results else results
    prompt = build_prompt(query, ranked, history=history)
    answer = generate(prompt, max_tokens=2048)
    sources = [normalize_source(r) for r in ranked]
    return {
        "answer": answer,
        "sources": [s.model_dump() for s in sources],
        "model": MODEL_NAME,
        "used_hybrid": use_hybrid,
        "used_rerank": use_rerank,
    }

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}

@app.get("/threads")
def list_threads():
    return get_threads()


@app.get("/threads/{thread_id}/messages")
def get_thread_messages_endpoint(thread_id: str):
    return get_thread_messages(thread_id)

@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest) -> QueryResponse:
    try:
        started = time.perf_counter()

        thread_id = ensure_thread(req.thread_id, title="New conversation")
        insert_message(thread_id, "user", req.query)

        recent_history = get_recent_messages(thread_id, limit=req.history_limit)

        cache_params = {
            "hybrid": req.hybrid,
            "rerank": req.rerank,
            "top_k_retrieve": req.top_k_retrieve,
            "top_k_rerank": req.top_k_rerank,
        }
        cache_key = make_cache_key(req.query, cache_params)
        cached = get_cached(cache_key)

        if cached:
            insert_message(thread_id, "user", req.query)
            insert_message(thread_id, "assistant", cached["answer"], sources=cached["sources"])
            return QueryResponse(
                thread_id=thread_id,
                answer=cached["answer"],
                sources=[normalize_source(s) for s in cached["sources"]],
                model=cached["model"],
                used_hybrid=req.hybrid,
                used_rerank=req.rerank,
            )


        result = ask(
            query=req.query,
            top_k_retrieve=req.top_k_retrieve,
            top_k_rerank=req.top_k_rerank,
            use_hybrid=req.hybrid,
            use_rerank=req.rerank,
            history=recent_history[:-1],
        )

        answer = result["answer"]
        sources = result["sources"]
        
        set_cached(cache_key, req.query, answer, sources, result["model"], cache_params)


        insert_message(thread_id, "assistant", answer, sources=sources)

        if req.thread_id is None:
            update_thread_title(thread_id, req.query[:60].strip())

        latency_ms = int((time.perf_counter() - started) * 1000)

        insert_rag_run(
            thread_id=thread_id,
            query=req.query,
            answer=answer,
            model=result["model"],
            latency_ms=latency_ms,
            used_hybrid=result["used_hybrid"],
            used_rerank=result["used_rerank"],
            params={
                "top_k_retrieve": req.top_k_retrieve,
                "top_k_rerank": req.top_k_rerank,
                "stream": req.stream,
                "history_limit": req.history_limit,
            },
            sources=sources,
        )

        return QueryResponse(thread_id=thread_id, **result)

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"RAG pipeline failed: {e}")