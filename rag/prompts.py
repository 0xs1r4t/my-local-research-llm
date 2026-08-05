def build_prompt(query: str, chunks: list[dict], history: list[dict] | None = None) -> str:
    history = history or []

    history_block = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in history
    )

    context = "\n\n---\n\n".join(
        f"[{c['source']}]\n{c['text']}" for c in chunks
    )

    return f"""<|im_start|>system
You are a research assistant helping with a dissertation on non-photorealistic rendering and augmented reality. Answer the question using only the context provided below. Be specific and cite which paper each point comes from. If the answer isn't in the context, say "I don't know based on the provided papers."<|im_end|>
<|im_start|>user
Conversation history:
{history_block}

Context:
{context}

Question: {query}<|im_end|>
<|im_start|>assistant
"""