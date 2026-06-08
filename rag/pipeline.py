import os

import psycopg2
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

# Local embedding model (nomic-embed-text or all-MiniLM-L6-v2)
embedder = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)

# Your llama.cpp model
llm = Llama(model_path=os.environ["LLM_MODEL_PATH"], n_gpu_layers=-1, n_ctx=16384)

conn = psycopg2.connect(os.environ["DATABASE_URL"])

def ingest(text: str, source: str, page: int = 0, section: str = ""):
    chunks = chunk_text(text, size=512, overlap=64)
    for chunk in chunks:
        vec = embedder.encode(chunk).tolist()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO dissertation_chunks (source, page, section, content, embedding) VALUES (%s,%s,%s,%s,%s)",
                (source, page, section, chunk, vec)
            )
    conn.commit()

def retrieve(query: str, top_k: int = 5):
    vec = embedder.encode(query).tolist()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT content, source, page, section,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM dissertation_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (vec, vec, top_k))
        return cur.fetchall()

def ask(query: str):
    chunks = retrieve(query)
    context = "\n\n".join([f"[{r[1]} p{r[2]}] {r[0]}" for r in chunks])
    prompt = f"""<|im_start|>system
You are a dissertation research assistant. Answer using ONLY the provided sources.
<|im_end|>
<|im_start|>user
Context:
{context}

Question: {query}
<|im_end|>
<|im_start|>assistant"""
    return llm(prompt, max_tokens=1024, stop=["<|im_end|>"])["choices"][0]["text"]