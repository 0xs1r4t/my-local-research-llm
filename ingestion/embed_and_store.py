import os, json
from pathlib import Path
from dotenv import load_dotenv
import psycopg2
from sentence_transformers import SentenceTransformer
from ingestion.chunk import by_heading, Chunk

load_dotenv()

EMBED_MODEL = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
DB_URL = os.environ["DB_URL"]
META = json.loads(Path("data/zotero_meta.json").read_text(encoding="utf-8")) if Path("data/zotero_meta.json").exists() else {}
os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ.get("HF_TOKEN", "")

embedder = SentenceTransformer(EMBED_MODEL)

def get_collections_for(source_filename: str) -> list[str]:
    for meta in META.values():
        path = meta.get("path", "")
        if source_filename in path or source_filename == Path(path).name:
            return meta.get("collections", [])
    return []

def setup_db(conn):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id SERIAL PRIMARY KEY,
                source TEXT,
                collections TEXT[],
                strategy TEXT,
                chunk_index INTEGER,
                content TEXT,
                embedding vector(384),
                ts_content TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
            );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING ivfflat (embedding vector_cosine_ops);")
        cur.execute("CREATE INDEX IF NOT EXISTS chunks_ts_idx ON chunks USING GIN (ts_content);")
    conn.commit()

def store_chunks(chunks: list[Chunk]):
    conn = psycopg2.connect(DB_URL)
    print(f" connected to {conn.get_dsn_parameters()}, embedding chunks...")
    setup_db(conn)
    texts = [c.text for c in chunks]
    embeddings = embedder.encode(texts, batch_size=32, show_progress_bar=True)
    with conn.cursor() as cur:
        for chunk, emb in zip(chunks, embeddings):
            cols = get_collections_for(chunk.source)
            cur.execute(
                "INSERT INTO chunks (source, collections, strategy, chunk_index, content, embedding) VALUES (%s,%s,%s,%s,%s,%s)",
                (chunk.source, cols, chunk.strategy, chunk.index, chunk.text, emb.tolist())
            )
    conn.commit()
    conn.close()
    print(f"  stored {len(chunks)} chunks")
    
def already_embedded(source_filename: str) -> bool:
    try:
        conn = psycopg2.connect(DB_URL)
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM chunks WHERE source = %s LIMIT 1", (source_filename,))
            exists = cur.fetchone() is not None
        conn.close()
        return exists
    except Exception:
        return False

if __name__ == "__main__":
    processed = Path("data/processed")
    for f in processed.glob("*.json"):
        doc = json.loads(f.read_text(encoding="utf-8"))
        chunks = by_heading(doc["content"], doc["source"])
        store_chunks(chunks)