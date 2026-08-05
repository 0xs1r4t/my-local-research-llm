import json
from pathlib import Path
import pymupdf4llm

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def parse_pdf(pdf_path: Path) -> dict:
    md_text = pymupdf4llm.to_markdown(str(pdf_path))
    return {"source": pdf_path.name, "content": md_text}

def parse_all():
    for pdf in RAW_DIR.glob("*.pdf"):
        out = OUT_DIR / pdf.with_suffix(".json").name
        if out.exists():
            print(f"  skip {pdf.name} (already parsed)")
            continue
        doc = parse_pdf(pdf)
        out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  parsed → {out.name}")

if __name__ == "__main__":
    parse_all()