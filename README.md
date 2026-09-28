# Semantic Retrieval & RAG

A compact research-paper assistant built to explore **semantic retrieval, reranking, grounded generation and retrieval evaluation**.

Instead of hiding the retrieval pipeline behind a large framework, the project keeps its main stages explicit: documents are chunked, embedded, indexed, retrieved and reranked before an answer is generated from the selected evidence.

## Why this project?

RAG systems are often demonstrated with a single successful question. This project focuses on the parts that determine whether the system is actually useful:

- How are documents split into retrievable units?
- Does semantic retrieval find the relevant evidence?
- Does reranking improve the initial ranking?
- Can the system refuse to answer when the evidence is insufficient?
- How can retrieval quality be measured independently from generation?

## Architecture

```text
PDF / text documents
       |
       v
  text extraction
       |
       v
    chunking
       |
       v
Sentence embeddings
       |
       v
   FAISS index
       |
query -> semantic retrieval -> cross-encoder reranking
                                  |
                                  v
                         grounded context
                                  |
                                  v
                           answer + sources
```

## Features

- Sentence-transformer embeddings
- FAISS vector search
- Cross-encoder reranking
- Source-aware context construction
- Evidence threshold for unsupported questions
- FastAPI endpoints for indexing, retrieval and answering
- Retrieval metrics: Recall@k and MRR
- Unit tests for core text-processing logic

## Tech stack

Python · FastAPI · Sentence Transformers · FAISS · Hugging Face · NumPy · Pydantic · Pytest

## Quick start

```bash
git clone https://github.com/amrabah/semantic-retrieval-rag.git
cd semantic-retrieval-rag
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API documentation is then available at `/docs`.

## Example workflow

Index local text documents:

```bash
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"documents":[{"source":"paper-a","text":"Retrieval augmented generation combines..."}]}'
```

Retrieve evidence:

```bash
curl "http://localhost:8000/search?q=What%20is%20RAG%3F&k=5"
```

## Evaluation

Retrieval is evaluated separately from answer generation. Given a small benchmark containing queries and relevant document/chunk identifiers, the evaluator reports:

- **Recall@k** — whether relevant evidence appears in the first k results.
- **MRR** — how early the first relevant result appears.

This separation makes it easier to diagnose whether a poor answer comes from retrieval or generation.

## Project status

This is a portfolio project under active development. The first version implements the retrieval/evaluation core; planned extensions include PDF ingestion, a configurable LLM provider, automated RAG evaluation and a lightweight UI.

## Author

**Aya Mrabah** — AI Engineer / Data Scientist

Interests: applied machine learning, NLP, explainable AI and robust AI systems.
