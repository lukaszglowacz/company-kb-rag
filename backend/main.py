import os

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from rag.chunker import TextChunker
from rag.embeddings import EmbeddingService
from rag.store import VectorStore

app = FastAPI(title="Company KB RAG API")

_store = VectorStore()
_chunker = TextChunker()


def get_embedding_service() -> EmbeddingService:
    return EmbeddingService(api_key=os.environ["OPENAI_API_KEY"])


class IngestRequest(BaseModel):
    text: str
    source: str


class IngestResponse(BaseModel):
    chunks_added: int
    total_chunks: int


@app.post("/ingest", response_model=IngestResponse)
def ingest(
    request: IngestRequest,
    embedding_service: EmbeddingService = Depends(get_embedding_service),
) -> IngestResponse:
    chunks = _chunker.chunk(request.text, request.source)
    embeddings = [embedding_service.get_embedding(c.text) for c in chunks]
    _store.add(chunks, embeddings)
    return IngestResponse(chunks_added=len(chunks), total_chunks=_store.count)


@app.get("/health")
def health() -> dict[str, str | int]:
    return {"status": "ok", "chunks_indexed": _store.count}
