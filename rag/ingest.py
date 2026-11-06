"""Document extraction and structuring: file -> pages -> chunks with metadata."""
import hashlib
import io
import re
from collections import Counter

from docx import Document
from pypdf import PdfReader

SUPPORTED = (".pdf", ".docx", ".txt", ".md")
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80

# "TERM = meaning" or "TERM: meaning", entries separated by ";" or newlines (glossaries, data dictionaries)
DEFINITION = re.compile(r"^\s*([A-Za-z][\w&/().\- ]{0,40}?)\s*(?:=|:)\s*(\S.{2,200})$")


def extract_pages(filename: str, data: bytes) -> list[dict]:
    """Return a list of {page, text} for the uploaded file."""
    name = filename.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return [{"page": i + 1, "text": p.extract_text() or ""} for i, p in enumerate(reader.pages)]
    if name.endswith(".docx"):
        doc = Document(io.BytesIO(data))
        parts = [p.text for p in doc.paragraphs]
        for table in doc.tables:
            for row in table.rows:
                parts.append(" | ".join(c.text for c in row.cells))
        return [{"page": 1, "text": "\n".join(parts)}]
    if name.endswith((".txt", ".md")):
        return [{"page": 1, "text": data.decode("utf-8", errors="ignore")}]
    raise ValueError(f"Unsupported file type: {filename}")


def _norm(line: str) -> str:
    """Normalise a line for repeat detection (digits ignored so 'Page 3' == 'Page 7')."""
    return re.sub(r"\d+", "#", line.strip().lower())


def remove_repeated_lines(pages: list[dict], edge: int = 3, ratio: float = 0.5) -> list[dict]:
    """Drop header/footer lines: lines in the first/last `edge` lines that repeat on > ratio of pages."""
    if len(pages) < 3:
        return pages
    counts = Counter()
    for p in pages:
        lines = [l for l in p["text"].splitlines() if l.strip()]
        counts.update({_norm(l) for l in lines[:edge] + lines[-edge:]})
    boiler = {k for k, v in counts.items() if v > ratio * len(pages)}
    for p in pages:
        p["text"] = "\n".join(l for l in p["text"].splitlines() if _norm(l) not in boiler)
    return pages


def clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_definitions(text: str) -> list[str]:
    """Pull 'TERM = meaning' style entries out as standalone sentences, e.g. 'NAV means Net Asset Value.'"""
    defs = []
    for line in re.split(r"[;\n]", text):
        m = DEFINITION.match(line)
        if m and not re.search(r"\d", m.group(1)):
            term, meaning = m.group(1).strip(), m.group(2).strip().rstrip(".")
            defs.append(f"{term} means {meaning}. ({term}: {meaning})")
    return defs


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into ~size-char chunks with overlap, preferring paragraph/sentence boundaries."""
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = text[start:end]
            cut = max(window.rfind("\n\n"), window.rfind(". "), window.rfind("\n"))
            if cut > size * 0.5:
                end = start + cut + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def process_document(filename: str, data: bytes) -> dict:
    """Structured record for one document: metadata + list of chunk records."""
    doc_id = hashlib.md5(data).hexdigest()[:12]
    pages = remove_repeated_lines(extract_pages(filename, data))
    chunks = []
    for p in pages:
        text = clean(p["text"])
        pieces = [("text", c) for c in chunk_text(text)] + [("definition", d) for d in extract_definitions(text)]
        for i, (kind, c) in enumerate(pieces):
            chunks.append({
                "id": f"{doc_id}-p{p['page']}-c{i}",
                "text": c,
                "metadata": {"doc_id": doc_id, "filename": filename, "page": p["page"], "chunk": i, "type": kind},
            })
    return {
        "doc_id": doc_id,
        "filename": filename,
        "num_pages": len(pages),
        "num_chunks": len(chunks),
        "chunks": chunks,
    }
