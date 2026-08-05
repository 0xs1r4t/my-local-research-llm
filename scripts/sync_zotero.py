import sys, shutil, sqlite3, json
from pathlib import Path

ZOTERO_DIR = Path("C:/Users/siria/Zotero")
DB_PATH    = ZOTERO_DIR / "zotero.sqlite"
STORAGE    = ZOTERO_DIR / "storage"
DST        = Path("data/raw")
META_FILE  = Path("data/zotero_meta.json")  # collection tags per PDF

def get_collection_map():
    """Returns {attachment_key: [collection_name, ...]} from zotero.sqlite"""
    # Copy DB first — Zotero locks it when open
    tmp_db = Path("data/zotero_tmp.sqlite")
    tmp_db.parent.mkdir(exist_ok=True)
    shutil.copy2(DB_PATH, tmp_db)

    conn = sqlite3.connect(tmp_db)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Get all PDF attachments and their parent item's collections
    cur.execute("""
        SELECT
            i.key                          AS attach_key,
            ia.path                        AS path,
            GROUP_CONCAT(c.collectionName, '||') AS collections
        FROM items i
        JOIN itemAttachments ia ON ia.itemID = i.itemID
        LEFT JOIN itemNotes ipar ON ipar.itemID = i.itemID
        LEFT JOIN collectionItems ci ON ci.itemID = COALESCE(ia.parentItemID, i.itemID)
        LEFT JOIN collections c ON c.collectionID = ci.collectionID
        WHERE ia.contentType = 'application/pdf'
        GROUP BY i.key
    """)

    result = {}
    for row in cur.fetchall():
        collections = [c.strip() for c in (row["collections"] or "").split("||") if c.strip()]
        result[row["attach_key"]] = {
            "path": row["path"],
            "collections": collections
        }

    conn.close()
    tmp_db.unlink()
    return result

def sync(filter_collections: list[str] | None = None):
    DST.mkdir(parents=True, exist_ok=True)
    cmap = get_collection_map()

    # Save full metadata for use during ingestion
    META_FILE.write_text(json.dumps(cmap, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved metadata for {len(cmap)} attachments → {META_FILE}")

    copied, skipped = 0, 0
    for key, meta in cmap.items():
        cols = meta["collections"]

        # Filter by collection if requested
        if filter_collections:
            if not any(f.lower() in [c.lower() for c in cols] for f in filter_collections):
                skipped += 1
                continue

        # Find the actual file
        pdf_path = STORAGE / key
        pdfs = list(pdf_path.glob("*.pdf")) if pdf_path.exists() else []
        if not pdfs:
            continue

        pdf = pdfs[0]
        target = DST / pdf.name
        if target.exists():
            skipped += 1
            continue

        shutil.copy2(pdf, target)
        print(f"  + [{', '.join(cols) or 'uncollected'}] {pdf.name}")
        copied += 1

    print(f"\nDone: {copied} copied, {skipped} skipped")

if __name__ == "__main__":
    # Pass collection names as args, or leave empty for all
    # e.g: python scripts/sync_zotero.py "relevant" "not completely irr"
    filter_cols = sys.argv[1:] if len(sys.argv) > 1 else None
    sync(filter_collections=filter_cols)