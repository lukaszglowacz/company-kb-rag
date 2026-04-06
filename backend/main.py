import os

import anthropic
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from rag.chunker import TextChunker
from rag.embeddings import EmbeddingService
from rag.pipeline import QueryResult, RAGPipeline
from rag.store import VectorStore

app = FastAPI(title="Company KB RAG API")

_store = VectorStore()
_chunker = TextChunker()


def get_embedding_service() -> EmbeddingService:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY environment variable not configured",
        )
    return EmbeddingService(api_key=api_key)


def get_anthropic_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="ANTHROPIC_API_KEY environment variable not configured",
        )
    return anthropic.Anthropic(api_key=api_key)


def get_pipeline(
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    anthropic_client: anthropic.Anthropic = Depends(get_anthropic_client),
) -> RAGPipeline:
    return RAGPipeline(
        store=_store,
        embedding_service=embedding_service,
        anthropic_client=anthropic_client,
    )


# ── Request / Response schemas ────────────────────────────────────────────────

class IngestRequest(BaseModel):
    text: str
    source: str


class IngestResponse(BaseModel):
    chunks_added: int
    total_chunks: int


class QueryRequest(BaseModel):
    question: str


class ChunkMetadataResponse(BaseModel):
    source: str
    score: float
    preview: str


class QueryResponse(BaseModel):
    answer: str
    retrieved_chunks: list[ChunkMetadataResponse]


class SummarizeRequest(BaseModel):
    text: str


class SummarizeResponse(BaseModel):
    summary: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/ingest", response_model=IngestResponse)
def ingest(
    request: IngestRequest,
    embedding_service: EmbeddingService = Depends(get_embedding_service),
) -> IngestResponse:
    chunks = _chunker.chunk(request.text, request.source)
    embeddings = [embedding_service.get_embedding(c.text) for c in chunks]
    _store.add(chunks, embeddings)
    return IngestResponse(chunks_added=len(chunks), total_chunks=_store.count)


@app.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> QueryResponse:
    result: QueryResult = pipeline.query(request.question)
    return QueryResponse(
        answer=result.answer,
        retrieved_chunks=[
            ChunkMetadataResponse(
                source=m.source, score=m.score, preview=m.preview
            )
            for m in result.retrieved_chunks
        ],
    )


@app.post("/summarize", response_model=SummarizeResponse)
def summarize(
    request: SummarizeRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> SummarizeResponse:
    return SummarizeResponse(summary=pipeline.summarize(request.text))


@app.get("/health")
def health() -> dict[str, str | int]:
    return {"status": "ok", "chunks_indexed": _store.count}
