from app.chunking import Chunk
from app.generation import build_context, grounded_answer
from app.retrieval import Hit

def test_context_contains_provenance():
    hits = [Hit(Chunk("p::chunk-0", "paper.pdf", "Evidence text"), 2.0)]
    context = build_context(hits)
    assert "paper.pdf" in context
    assert "Evidence text" in context

def test_refuses_without_evidence():
    answer, supported = grounded_answer("question", [])
    assert supported is False
    assert "enough evidence" in answer
