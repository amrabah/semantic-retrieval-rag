import pytest
from app.chunking import chunk_text

def test_empty_text_returns_no_chunks():
    assert chunk_text("", "paper") == []

def test_chunk_ids_preserve_source():
    chunks = chunk_text("one two three four five six", "paper-a", chunk_size=4, overlap=1)
    assert chunks[0].chunk_id == "paper-a::chunk-0"
    assert all(c.source == "paper-a" for c in chunks)

def test_invalid_overlap_is_rejected():
    with pytest.raises(ValueError):
        chunk_text("some text", "x", chunk_size=10, overlap=10)
