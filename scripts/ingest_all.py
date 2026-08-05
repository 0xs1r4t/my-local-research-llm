import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from itertools import islice
from ingestion.parse_pdfs import parse_pdf
from ingestion.chunk import by_sentence
from ingestion.embed_and_store import store_chunks, already_embedded

PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)

def batched(iterable, n):
    it = iter(iterable)
    while chunk := list(islice(it, n)):
        yield chunk

def ingest_batch(pdfs: list[Path]):
    for pdf in pdfs:
        out = PROCESSED / pdf.with_suffix(".json").name

        # Parse if needed
        if not out.exists():
            print(f"  parsing  {pdf.name}")
            doc = parse_pdf(pdf)
            out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            print(f"  skip parse {pdf.name} (json exists)")

        # Embed if needed — check DB, not filesystem
        if already_embedded(pdf.name):
            print(f"  skip embed {pdf.name} (already in DB)")
            continue

        doc = json.loads(out.read_text(encoding="utf-8"))
        chunks = by_sentence(doc["content"], doc["source"])
        print(f"  embedding {len(chunks)} chunks from {pdf.name}")
        store_chunks(chunks)

def main(batch_size: int = 10):
    all_pdfs = sorted(Path("data/raw").glob("*.pdf"))
    total = len(all_pdfs)
    print(f"\n{total} PDFs found, ingesting in batches of {batch_size}\n")
    for i, batch in enumerate(batched(all_pdfs, batch_size), 1):
        print(f"── Batch {i} / {-(-total // batch_size)} ──")
        ingest_batch(batch)
        print()
    print("All done.")

if __name__ == "__main__":
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    main(batch_size=size)