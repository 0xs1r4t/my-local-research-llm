import re
from dataclasses import dataclass

@dataclass
class Chunk:
    text: str
    source: str
    strategy: str
    index: int

def by_heading(text: str, source: str) -> list[Chunk]:
    """Split on Markdown headings (##, ###)."""
    parts = re.split(r"(?=^#{1,3} )", text, flags=re.MULTILINE)
    return [
        Chunk(text=p.strip(), source=source, strategy="heading", index=i)
        for i, p in enumerate(parts) if len(p.strip()) > 50
    ]

def by_fixed(text: str, source: str, size: int = 400, overlap: int = 80) -> list[Chunk]:
    """Fixed-size word chunks with overlap."""
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i : i + size])
        chunks.append(Chunk(text=chunk, source=source, strategy="fixed", index=len(chunks)))
        i += size - overlap
    return chunks

def by_sentence(text: str, source: str, max_words: int = 300) -> list[Chunk]:
    """Sentence-aware: accumulate sentences until max_words."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current, word_count = [], [], 0
    for sent in sentences:
        wc = len(sent.split())
        if word_count + wc > max_words and current:
            chunks.append(Chunk(
                text=" ".join(current), source=source,
                strategy="sentence", index=len(chunks)
            ))
            current, word_count = [], 0
        current.append(sent)
        word_count += wc
    if current:
        text = " ".join(current).strip()
        if len(text.split()) >= 30:  # avoid tiny last chunk
            chunks.append(Chunk(
                text=text, source=source,
                strategy="sentence", index=len(chunks)
            ))
    return chunks

def by_heading_then_sentence(text: str, source: str, max_words: int = 300) -> list[Chunk]:
    import re
    sections = re.split(r"(?=^#{1,3} )", text, flags=re.MULTILINE)
    all_chunks = []
    for section in sections:
        if len(section.split()) <= max_words:
            if section.strip():
                all_chunks.append(Chunk(text=section.strip(), source=source, strategy="hybrid", index=len(all_chunks)))
        else:
            # section too big, split by sentence
            sub = by_sentence(section, source, max_words=max_words)
            for c in sub:
                all_chunks.append(Chunk(text=c.text, source=source, strategy="hybrid", index=len(all_chunks)))
    return all_chunks