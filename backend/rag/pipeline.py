from dataclasses import dataclass

import anthropic

from rag.chunker import Chunk
from rag.embeddings import EmbeddingService
from rag.store import VectorStore

CLAUDE_MODEL = "claude-sonnet-4-20250514"
TOP_K_CHUNKS = 5
CHUNK_PREVIEW_LENGTH = 100
CHUNK_NUMBERING_OFFSET = 1  # convert 0-based index to 1-based display
MAX_TOKENS_ANSWER = 1024
MAX_TOKENS_SUMMARY = 64

SYSTEM_PROMPT = (
    "You are a company knowledge base assistant. "
    "Answer questions using ONLY the provided document excerpts. "
    "If the answer is not in the excerpts, say so clearly. "
    "Do not invent information."
)

SUMMARIZE_PROMPT = (
    "Summarize this question in maximum 12 words, "
    "in the same language as the input. "
    "Return only the summary, no punctuation."
)


@dataclass(frozen=True)
class ChunkMetadata:
    source: str
    score: float
    preview: str


@dataclass(frozen=True)
class QueryResult:
    answer: str
    retrieved_chunks: list[ChunkMetadata]


class RAGPipeline:
    """Orchestrates retrieval and answer generation.

    Single Responsibility: prompt construction and LLM calls only.
    Dependency Inversion: VectorStore and EmbeddingService injected.
    """

    def __init__(
        self,
        store: VectorStore,
        embedding_service: EmbeddingService,
        anthropic_client: anthropic.Anthropic,
    ) -> None:
        self._store = store
        self._embedding_service = embedding_service
        self._client = anthropic_client

    def build_prompt(self, question: str, chunks: list[Chunk]) -> str:
        if not chunks:
            return (
                f"No relevant documents found.\n\nQuestion: {question}"
            )
        excerpts = "\n\n".join(
            f"[{i + CHUNK_NUMBERING_OFFSET}] (source: {c.source})\n{c.text}"
            for i, c in enumerate(chunks)
        )
        return f"Document excerpts:\n\n{excerpts}\n\nQuestion: {question}"

    def _call_llm(
        self,
        prompt: str,
        max_tokens: int,
        system: str | None = None,
    ) -> str:
        if system is not None:
            message = self._client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
        else:
            message = self._client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
        if not message.content:
            raise ValueError("Empty response from Claude API")
        block = message.content[0]
        if not isinstance(block, anthropic.types.TextBlock):
            raise ValueError("Unexpected non-text response from Claude API")
        return block.text

    def query(self, question: str) -> QueryResult:
        query_embedding = self._embedding_service.get_embedding(question)
        results = self._store.search(query_embedding, top_k=TOP_K_CHUNKS)

        chunks = [chunk for chunk, _ in results]
        prompt = self.build_prompt(question, chunks)
        answer = self._call_llm(prompt, MAX_TOKENS_ANSWER, system=SYSTEM_PROMPT)

        metadata = [
            ChunkMetadata(
                source=chunk.source,
                score=score,
                preview=(chunk.text or "")[:CHUNK_PREVIEW_LENGTH],
            )
            for chunk, score in results
        ]
        return QueryResult(answer=answer, retrieved_chunks=metadata)

    def summarize(self, text: str) -> str:
        return self._call_llm(
            f"{SUMMARIZE_PROMPT}\n\n{text}",
            MAX_TOKENS_SUMMARY,
        )
