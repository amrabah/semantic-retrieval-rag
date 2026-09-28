from __future__ import annotations
from collections.abc import Callable
from .retrieval import Hit

def build_context(hits: list[Hit]) -> str:
    return "\n\n".join(
        f"[{i}] Source: {hit.chunk.source}\n{hit.chunk.text}"
        for i, hit in enumerate(hits, start=1)
    )

def grounded_answer(
    question: str,
    hits: list[Hit],
    generator: Callable[[str], str] | None = None,
    min_score: float = 0.0,
) -> tuple[str, bool]:
    """Generate only when retrieval supplies sufficient evidence.

    A generator is injected deliberately so the retrieval stack is independent
    from any single commercial or local LLM provider.
    """
    if not hits or hits[0].score < min_score:
        return "I do not have enough evidence in the indexed documents to answer this question.", False

    context = build_context(hits)
    if generator is None:
        return (
            "Relevant evidence was retrieved. Configure an LLM generator to produce a grounded answer.",
            True,
        )

    prompt = f"""Answer the question using only the evidence below.
If the evidence is insufficient, say that you do not have enough evidence.
Cite evidence using [1], [2], ... and do not invent sources.

Question: {question}

Evidence:
{context}

Answer:"""
    return generator(prompt), True
