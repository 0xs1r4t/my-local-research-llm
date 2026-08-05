from ingestion.chunk import by_heading, by_sentence, by_fixed
import json
from pathlib import Path

doc = json.loads(Path("data/processed/Kerbl et al. - 2023 - 3D Gaussian Splatting for Real-Time Radiance Field Rendering.json").read_text(encoding="utf-8"))
text = doc["content"]

for strategy, chunks in [("heading", by_heading(text, "test")), ("sentence", by_sentence(text, "test")), ("fixed", by_fixed(text, "test"))]:
    sizes = [len(c.text.split()) for c in chunks]
    print(f"{strategy:10} → {len(chunks)} chunks, avg {sum(sizes)//len(sizes)} words, max {max(sizes)} words")