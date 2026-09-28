from dataclasses import dataclass

@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    text: str

def chunk_text(text: str, source: str, chunk_size: int = 180, overlap: int = 30) -> list[Chunk]:
    """Split text into overlapping word windows while preserving provenance."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size.")
    words = text.split()
    if not words:
        return []
    step = chunk_size - overlap
    chunks = []
    for i, start in enumerate(range(0, len(words), step)):
        part = words[start:start + chunk_size]
        if not part:
            break
        chunks.append(Chunk(f"{source}::chunk-{i}", source, " ".join(part)))
        if start + chunk_size >= len(words):
            break
    return chunks
