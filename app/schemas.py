from pydantic import BaseModel, Field

class Document(BaseModel):
    source: str
    text: str

class IndexRequest(BaseModel):
    documents: list[Document]

class SearchResult(BaseModel):
    source: str
    chunk_id: str
    text: str
    score: float

class AnswerResponse(BaseModel):
    answer: str
    supported: bool
    sources: list[SearchResult] = Field(default_factory=list)
