"""
rag.py - Retrieval-Augmented Generation: the "retrieval" half.

1. Chunk:  split each manual into sections (one chunk per "## " heading)
2. Embed:  turn each chunk into a vector with the embedding model
3. Index:  save vectors to data/index.json (a simple local vector store)
4. Search: embed the question, return the chunks with the highest cosine similarity

Locally we use this small vector store; in Azure it is replaced by Azure AI Search.
"""

import json
import math
from pathlib import Path

from tools.llm import embed

ROOT = Path(__file__).resolve().parent.parent
MANUALS_DIR = ROOT / "data" / "manuals"
INDEX_PATH = ROOT / "data" / "index.json"


def chunk_markdown(path: Path) -> list[dict]:
    """Structure-aware chunking: one chunk per '## ' section, prefixed with the doc title."""
    lines = path.read_text().splitlines()
    title = lines[0].lstrip("# ").strip() if lines else path.stem
    chunks, heading, body = [], None, []

    def flush():
        text = "\n".join(body).strip()
        if heading and text:
            chunks.append({
                "id": f"{path.name}#{heading}",
                "source": path.name,
                "section": heading,
                "text": f"{title} - {heading}\n{text}",
            })

    for line in lines[1:]:
        if line.startswith("## "):
            flush()
            heading, body = line[3:].strip(), []
        else:
            body.append(line)
    flush()
    return chunks


def build_index() -> list[dict]:
    chunks = [c for p in sorted(MANUALS_DIR.glob("*.md")) for c in chunk_markdown(p)]
    vectors = embed([c["text"] for c in chunks])  # one API call for all chunks (batching)
    for chunk, vector in zip(chunks, vectors):
        chunk["vector"] = vector
    INDEX_PATH.write_text(json.dumps(chunks))
    return chunks


def load_index() -> list[dict]:
    if not INDEX_PATH.exists():
        return build_index()
    return json.loads(INDEX_PATH.read_text())


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def search(query: str, k: int = 3) -> list[dict]:
    """Return the k chunks most similar in meaning to the query."""
    index = load_index()
    q = embed([query])[0]
    scored = sorted(index, key=lambda c: cosine(q, c["vector"]), reverse=True)[:k]
    return [{**{key: c[key] for key in ("id", "source", "section", "text")},
             "score": round(cosine(q, c["vector"]), 3)} for c in scored]


if __name__ == "__main__":
    chunks = build_index()
    print(f"Indexed {len(chunks)} chunks from {MANUALS_DIR.relative_to(ROOT)}/\n")
    for question in ["product temperature rising at the filler",
                     "vibration climbing on the main drive"]:
        print(f"Query: {question}")
        for hit in search(question):
            print(f"  {hit['score']}  {hit['id']}")
        print()
