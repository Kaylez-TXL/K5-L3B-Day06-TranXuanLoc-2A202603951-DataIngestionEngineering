"""Page-local chunks with stable IDs and exact source character offsets."""
import hashlib
import json
import re
from pathlib import Path
from .models import Chunk, DocumentPage


def _chunk(page: DocumentPage, start: int, end: int, method: str) -> Chunk:
    # Preserve exact source slices so evidence and coherence anchors stay traceable.
    key = f"{page.document_id}:{page.page}:{method}:{start}:{end}:{page.text[start:end]}"
    return Chunk(
        chunk_id=hashlib.sha256(key.encode("utf-8")).hexdigest(),
        document_id=page.document_id, source=page.source, page=page.page,
        text=page.text[start:end],
        metadata={**page.metadata, "char_start": start, "char_end": end, "chunking": method},
    )


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Use character windows; the last window stops at the end of the page."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Require size > 0 and 0 <= overlap < size.")
    chunks = []
    start = 0
    while start < len(page.text):
        end = min(start + size, len(page.text))
        if page.text[start:end].strip():
            chunks.append(_chunk(page, start, end, "fixed_size"))
        if end == len(page.text):
            break
        start += size - overlap
    return chunks


def _blocks(text: str) -> list[tuple[int, int]]:
    """Infer boundaries from blank lines, headings, bullets and sentence endings.

    PDF extraction supplies visual line wraps, not reliable paragraph boundaries.
    Keep wrapped lines together unless a sentence ends; do not claim recovery of
    the original Word paragraphs. Headings remain attached to following content.
    """
    spans = []
    start = None
    previous = ""
    heading = re.compile(r"^(?:Chương\s+\d|\d+(?:\.\d+)+\.?\s|[-•]\s)")
    for match in re.finditer(r"[^\n]+(?:\n|$)|\n", text):
        line = match.group().strip()
        if not line:
            if start is not None:
                spans.append((start, match.start()))
                start = None
            previous = ""
            continue
        if start is not None and (heading.match(line) or previous.endswith((".", "!", "?", ";"))):
            spans.append((start, match.start()))
            start = None
        if start is None:
            start = match.start()
        previous = line
    if start is not None:
        spans.append((start, len(text)))
    return spans


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Pack inferred structural blocks; split oversized blocks at whitespace.

    No overlap, no cross-page chunks. A long word may require a hard character
    cut. Boundaries and limitations must be included in the experiment report.
    """
    if max_size <= 0:
        raise ValueError("max_size must be positive.")
    pieces = []
    for start, end in _blocks(page.text):
        while end - start > max_size:
            limit = start + max_size
            spaces = list(re.finditer(r"\s+", page.text[start:limit]))
            cut = start + spaces[-1].end() if spaces else limit
            if cut <= start:
                cut = limit
            pieces.append((start, cut))
            start = cut
        if end > start:
            pieces.append((start, end))
    packed = []
    for start, end in pieces:
        if packed and end - packed[-1][0] <= max_size:
            packed[-1] = (packed[-1][0], end)
        else:
            packed.append((start, end))
    return [_chunk(page, start, end, "by_paragraph") for start, end in packed if page.text[start:end].strip()]


if __name__ == "__main__":
    path = Path(__file__).resolve().parents[2] / "data/processed/pages.jsonl"
    pages = [DocumentPage(**{key: row[key] for key in ("document_id", "source", "page", "text", "metadata")})
             for row in map(json.loads, path.read_text(encoding="utf-8").splitlines())]
    for name, split in [("A", lambda p: fixed_size(p, 500, 50)), ("B/C", lambda p: by_paragraph(p, 500))]:
        chunks = [chunk for page in pages for chunk in split(page)]
        lengths = [len(chunk.text) for chunk in chunks]
        print(f"{name}: {len(chunks)} chunks; characters min/mean/max = "
              f"{min(lengths)}/{sum(lengths)/len(lengths):.1f}/{max(lengths)}")
    print("Preview only. No embeddings, database writes or retrieval metrics yet.")
