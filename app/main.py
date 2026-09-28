from fastapi import FastAPI, HTTPException, Query
from .chunking import chunk_text
from .generation import grounded_answer
from .retrieval import SemanticRetriever
from .schemas import AnswerResponse, IndexRequest, SearchResult

app = FastAPI(
    title="Semantic Retrieval & RAG",
    description="Dense retrieval, reranking and grounded-answering portfolio API.",
    version="0.1.0",
)

retriever: SemanticRetriever | None = None

def _results(hits):
    return [
        SearchResult(
            source=h.chunk.source,
            chunk_id=h.chunk.chunk_id,
            text=h.chunk.text,
            score=h.score,
        )
        for h in hits
    ]

@app.get("/health")
def health():
    return {"status": "ok", "indexed": retriever is not None}

@app.post("/index")
def index_documents(payload: IndexRequest):
    global retriever
    chunks = []
    for doc in payload.documents:
        chunks.extend(chunk_text(doc.text, doc.source))
    if not chunks:
        raise HTTPException(400, "No indexable text was provided.")
    retriever = SemanticRetriever()
    retriever.build(chunks)
    return {"documents": len(payload.documents), "chunks": len(chunks)}

@app.get("/search", response_model=list[SearchResult])
def search(q: str = Query(min_length=2), k: int = Query(5, ge=1, le=20)):
    if retriever is None:
        raise HTTPException(409, "Index documents before searching.")
    return _results(retriever.search(q, k=k))

@app.get("/answer", response_model=AnswerResponse)
def answer(q: str = Query(min_length=2), k: int = Query(5, ge=1, le=20)):
    if retriever is None:
        raise HTTPException(409, "Index documents before answering.")
    hits = retriever.search(q, k=k)
    text, supported = grounded_answer(q, hits)
    return AnswerResponse(answer=text, supported=supported, sources=_results(hits))
