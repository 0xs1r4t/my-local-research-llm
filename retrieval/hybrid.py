import os
from functools import lru_cache
from typing import Optional

import psycopg2
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


load_dotenv()

embedder = SentenceTransformer(
    os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
)


def _pretty_title(source: str | None) -> str:
    if not source:
        return "unknown"

    name = os.path.basename(source)
    name = os.path.splitext(name)[0]
    name = name.replace("_", " ").replace("-", " ")
    name = " ".join(name.split())
    return name.title() if name else "unknown"

@lru_cache(maxsize=512)
def _embed(query: str) -> tuple:
    return tuple(embedder.encode(query).tolist())

def search(
    query: str,
    top_k: int = 10,
    collections: Optional[list[str]] = None,
    hybrid: bool = True,
) -> list[dict]:
    conn = psycopg2.connect(os.environ["DB_URL"])
    emb = list(_embed(query))  # unhash back to list for psycopg2
    results: dict[int, dict] = {}

    try:
        with conn.cursor() as cur:
            semantic_where = []
            semantic_params = []

            if collections:
                semantic_where.append("collections && %s::text[]")
                semantic_params.append(collections)

            semantic_where_sql = (
                f"WHERE {' AND '.join(semantic_where)}"
                if semantic_where else ""
            )

            semantic_sql = f"""
                SELECT
                    id,
                    source,
                    chunk_index,
                    content,
                    1 - (embedding <=> %s::vector) AS score
                FROM chunks
                {semantic_where_sql}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """

            if collections:
                semantic_exec_params = [emb] + semantic_params + [emb, top_k]
            else:
                semantic_exec_params = [emb, emb, top_k]

            cur.execute(semantic_sql, semantic_exec_params)

            for row in cur.fetchall():
                row_id, source, chunk_index, content, score = row
                results[row_id] = {
                    "id": row_id,
                    "source": source,
                    "title": _pretty_title(source),
                    "text": content,
                    "score": float(score),
                    "chunk_id": f"{source}#chunk-{chunk_index}" if source is not None and chunk_index is not None else str(row_id),
                    "metadata": {
                        "raw_source": source,
                        "chunk_id": f"{source}#chunk-{chunk_index}" if source is not None and chunk_index is not None else str(row_id),
                        "chunk_index": chunk_index,
                    },
                }

            if not hybrid:
                return sorted(
                    results.values(),
                    key=lambda x: x["score"],
                    reverse=True,
                )[:top_k]

            fulltext_where = ["ts_content @@ plainto_tsquery('english', %s)"]
            fulltext_params = [query]

            if collections:
                fulltext_where.append("collections && %s::text[]")
                fulltext_params.append(collections)

            fulltext_sql = f"""
                SELECT
                    id,
                    source,
                    chunk_index,
                    content,
                    ts_rank(ts_content, plainto_tsquery('english', %s)) AS score
                FROM chunks
                WHERE {' AND '.join(fulltext_where)}
                ORDER BY score DESC
                LIMIT %s
            """

            fulltext_exec_params = [query] + fulltext_params + [top_k]
            cur.execute(fulltext_sql, fulltext_exec_params)

            for row in cur.fetchall():
                row_id, source, chunk_index, content, ft_score = row
                ft_score = float(ft_score)

                if row_id not in results:
                    results[row_id] = {
                        "id": row_id,
                        "source": source,
                        "title": _pretty_title(source),
                        "text": content,
                        "score": ft_score,
                        "chunk_id": f"{source}#chunk-{chunk_index}" if source is not None and chunk_index is not None else str(row_id),
                        "metadata": {
                            "raw_source": source,
                            "chunk_id": f"{source}#chunk-{chunk_index}" if source is not None and chunk_index is not None else str(row_id),
                            "chunk_index": chunk_index,
                        },
                    }
                else:
                    results[row_id]["score"] += ft_score

        return sorted(
            results.values(),
            key=lambda x: x["score"],
            reverse=True,
        )[:top_k]

    finally:
        conn.close()