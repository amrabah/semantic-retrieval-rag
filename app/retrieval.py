from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer
from .chunking import Chunk

@dataclass
class Hit:
    chunk: Chunk
    score: float

class SemanticRetriever:
    """Two-stage dense retrieval + cross-encoder reranking."""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        import faiss
        self.faiss = faiss
        self.encoder = SentenceTransformer(embedding_model)
        self.reranker = CrossEncoder(reranker_model)
        self.index = None
        self.chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build an index without chunks.")
        vectors = self.encoder.encode(
            [c.text for c in chunks], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")
        self.index = self.faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.chunks = list(chunks)

    def search(self, query: str, k: int = 5, candidates: int = 20) -> list[Hit]:
        if self.index is None:
            raise RuntimeError("Index has not been built.")
        n = min(max(k, candidates), len(self.chunks))
        q = self.encoder.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        _, ids = self.index.search(q, n)
        candidates_ = [self.chunks[i] for i in ids[0] if i >= 0]
        scores = self.reranker.predict([(query, c.text) for c in candidates_])
        ranked = sorted(zip(candidates_, scores), key=lambda x: float(x[1]), reverse=True)
        return [Hit(c, float(s)) for c, s in ranked[:k]]
