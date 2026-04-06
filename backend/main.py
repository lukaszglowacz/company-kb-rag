import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import anthropic
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from data.documents import load_documents
from rag.chunker import TextChunker
from rag.embeddings import EmbeddingService
from rag.pipeline import RAGPipeline
from rag.store import VectorStore

logger = logging.getLogger(__name__)

_store = VectorStore()
_chunker = TextChunker()


def _ingest_documents(embedding_service: EmbeddingService) -> None:
    docs = load_documents()
    for text, source in docs:
        chunks = _chunker.chunk(text, source)
        embeddings = [embedding_service.get_embedding(c.text) for c in chunks]
        _store.add(chunks, embeddings)
    logger.info("Auto-ingestion complete: %d chunks indexed", _store.count)


def _get_openai_api_key() -> str | None:
    return os.environ.get("OPENAI_API_KEY")


def _get_anthropic_api_key() -> str | None:
    return os.environ.get("ANTHROPIC_API_KEY")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    api_key = _get_openai_api_key()
    if api_key:
        try:
            _ingest_documents(EmbeddingService(api_key=api_key))
        except Exception:
            logger.exception(
                "Auto-ingestion failed — continuing without pre-loaded docs"
            )
    else:
        logger.warning("OPENAI_API_KEY not set — skipping auto-ingestion")
    yield


app = FastAPI(title="Company KB RAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_embedding_service() -> EmbeddingService:
    api_key = _get_openai_api_key()
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY environment variable not configured",
        )
    return EmbeddingService(api_key=api_key)


def get_anthropic_client() -> anthropic.Anthropic:
    api_key = _get_anthropic_api_key()
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
    result = pipeline.query(request.question)
    return QueryResponse(
        answer=result.answer,
        retrieved_chunks=[
            ChunkMetadataResponse(
                source=m.source, score=m.score, preview=m.preview
            )
            for m in result.retrieved_chunks
        ],
    )


@app.post("/query/stream")
def query_stream(
    request: QueryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> StreamingResponse:
    return StreamingResponse(
        pipeline.stream_query(request.question),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
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
