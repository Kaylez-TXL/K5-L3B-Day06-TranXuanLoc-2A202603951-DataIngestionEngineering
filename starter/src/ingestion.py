"""Extract physical PDF pages without losing their source locations."""
import hashlib
import json
import logging
import re
import unicodedata
from pathlib import Path
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pypdf import PdfReader
from .models import DocumentPage

logger = logging.getLogger(__name__)

def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    """Keep every physical page, including empty pages; log extraction failures."""
    pages = []
    for number, page in enumerate(PdfReader(path).pages, start=1):
        metadata = {}
        try:
            text = normalize_text(page.extract_text() or "")
        except Exception as exc:
            logger.exception("Cannot extract %s page %s", document_id, number)
            text = ""
            metadata["extraction_error"] = type(exc).__name__
        if not text:
            logger.warning("No extracted text: %s page %s", document_id, number)
        pages.append(DocumentPage(document_id, source, number, text, metadata))
    return pages


def normalize_text(text: str) -> str:
    """Normalize NFC and horizontal whitespace, retaining lines and paragraphs.

    Do not guess Vietnamese word repairs or rewrite mathematical symbols.
    """
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00ad", "").replace("\u200b", "")
    lines = [re.sub(r"[^\S\n]+", " ", line).strip() for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()

REPO = Path(__file__).resolve().parents[2]

DOCUMENT_ID = "auag-thesis-2026"
SOURCE = "data/raw/22022101_TranXuanLoc.pdf"
CHAPTERS = [
    (12, 13, 1, "Mở đầu"),
    (14, 21, 2, "Tổng quan về cấu trúc lõi-vỏ và đặc trưng quang học của hạt nano kim loại"),
    (22, 37, 3, "Cơ sở lý thuyết và phương pháp nghiên cứu"),
    (38, 64, 4, "Kết quả và thảo luận"),
    (65, 65, 5, "Kết luận"),
]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare_corpus():
    pdf = REPO / SOURCE
    if not pdf.is_file():
        raise SystemExit(f"Missing corpus: {pdf}. See README.md.")
    pages = load_pdf(pdf, DOCUMENT_ID, SOURCE)
    if len(pages) != 70:
        raise SystemExit("Expected the inspected 70-page PDF; review page ranges for a different version.")
    records, excluded, warnings = [], [], []
    for page in pages:
        chapter = next((c for c in CHAPTERS if c[0] <= page.page <= c[1]), None)
        if chapter is None:
            excluded.append({"page": page.page, "reason": "front_matter" if page.page < 12 else "references"})
            continue
        # Printed page numbers are alone at the start of the extracted body pages.
        page.text = re.sub(rf"^\s*{page.page}\s*\n", "", page.text).strip()
        page.metadata.update({"chapter": chapter[2], "chapter_title": chapter[3], "language": "vi"})
        row = asdict(page)
        row["page_id"] = f"{DOCUMENT_ID}:p{page.page:03d}"
        row["char_count"] = len(page.text)
        if not page.text or "extraction_error" in page.metadata:
            raise SystemExit(f"Body page {page.page} could not be extracted. Review before counting it.")
        if len(page.text) < 600:
            warnings.append({"page": page.page, "kind": "short_text_visual_review", "char_count": len(page.text),
                             "note": "Retained. May be figure-heavy; extracted text does not describe graph pixels."})
        records.append(row)
    if len(records) < 30:
        raise SystemExit("Fewer than 30 extracted body pages.")
    processed = REPO / "data/processed/pages.jsonl"
    processed.parent.mkdir(parents=True, exist_ok=True)
    processed.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    write_json(REPO / "data/manifest.json", {
        "source_id": DOCUMENT_ID,
        "title": "Nghiên cứu quang phổ của hạt nano lõi–vỏ Au@Ag và ứng dụng học máy trong dự đoán đặc tính quang học",
        "author": "Trần Xuân Lộc", "source_path_or_url": SOURCE,
        "license_or_permission": "Tôi là Trần Xuân Lộc, tác giả đồ án này và người thực hiện bài thực hành Data Ingestion Engineering. Tôi sử dụng đồ án của mình làm corpus cho bài nộp; các tài liệu và hình ảnh tham khảo vẫn giữ nguồn trích dẫn trong đồ án.",
        "accessed_at": datetime.now(timezone(timedelta(hours=7))).isoformat(),
        "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        "physical_page_count": len(pages), "included_page_count": len(records),
        "counting_rule": "Toàn bộ phần nội dung chính, trang PDF vật lý 12–65; giữ trang có hình/bảng. Loại trang đầu 1–11 và tài liệu tham khảo 66–70. Không áp ngưỡng 1500 ký tự của nguồn không phân trang lên PDF.",
        "pages": [{"page_id": r["page_id"], "page": r["page"], "char_count": r["char_count"], "chapter": r["metadata"]["chapter"]} for r in records],
        "excluded_pages": excluded,
        "metadata_provenance": {"chapter": "Chapter headings and table of contents checked against physical page boundaries. Integer 1–5.", "language": "Vietnamese source text; vi."},
        "normalization": "Unicode NFC, horizontal whitespace, remove soft hyphens/zero-width spaces and standalone leading printed page number; preserve line breaks. No guessed word or equation repair.",
        "limitations": ["Text extraction does not interpret figures.", "PDF line breaks are not reliable paragraph boundaries.", "Some Vietnamese words contain internal extraction spaces.", "Repeated section number 4.1.1; use chapter as the initial filter, not section number as a unique ID."],
    })
    write_json(REPO / "results/corpus_audit.json", {
        "document_id": DOCUMENT_ID, "included_pages": len(records),
        "normalized_characters": sum(r["char_count"] for r in records),
        "empty_body_pages": 0, "warnings": warnings,
        "status": "Text extraction checked; figures/tables and benchmark evidence require visual/manual verification.",
    })
    print(f"Prepared {len(records)} body pages; {sum(r['char_count'] for r in records)} characters; {len(warnings)} short-page warnings.")


if __name__ == "__main__":
    prepare_corpus()
