import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from ingestion.chunk import by_heading, by_sentence, by_fixed, by_heading_then_sentence

# Pick 3 representative papers from your corpus
test_files = [
    "Kerbl et al. - 2023 - 3D Gaussian Splatting for Real-Time Radiance Field Rendering.json",
    "Chen et al. - 2012 - A non-photorealistic rendering framework with temporal coherence for augmented reality.json",
    "Bridson - SIGGRAPH 2007 Course Notes.json",
]

strategies = {
    "heading":         by_heading,
    "sentence":        by_sentence,
    "fixed":           by_fixed,
    "hybrid":          by_heading_then_sentence,
}

for fname in test_files:
    path = Path("data/processed") / fname
    if not path.exists():
        print(f"missing: {fname}\n"); continue
    doc = json.loads(path.read_text(encoding="utf-8"))
    text = doc["content"]
    print(f"\n{'─'*60}")
    print(f"📄 {fname[:55]}")
    print(f"{'─'*60}")
    for name, fn in strategies.items():
        chunks = fn(text, fname)
        sizes = [len(c.text.split()) for c in chunks]
        avg = sum(sizes) // len(sizes)
        print(f"  {name:10} → {len(chunks):3} chunks | avg {avg:4}w | max {max(sizes):4}w | min {min(sizes):3}w")