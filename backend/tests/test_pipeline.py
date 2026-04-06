import json
import pytest
from unittest.mock import MagicMock

import anthropic

from rag.chunker import Chunk
from rag.embeddings import EmbeddingService
from rag.pipeline import (
    CHUNK_PREVIEW_LENGTH,
    RAGPipeline,
    QueryResult,
)
from rag.store import VectorStore


def _make_streaming_client_mock(tokens: list[str]) -> MagicMock:
    mock_stream = MagicMock()
    mock_stream.text_stream = iter(tokens)
    mock_stream.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream.__exit__ = MagicMock(return_value=False)
    mock_client = MagicMock()
    mock_client.messages.stream.return_value = mock_stream
    return mock_client


def _make_streaming_pipeline(store: VectorStore, tokens: list[str]) -> RAGPipeline:
    embedding_service = MagicMock(spec=EmbeddingService)
    embedding_service.get_embedding.side_effect = EmbeddingService.mock_embedding
    return RAGPipeline(
        store=store,
        embedding_service=embedding_service,
        anthropic_client=_make_streaming_client_mock(tokens),
    )


def _make_client_mock(text: str = "Mocked answer") -> MagicMock:
    mock_text_block = MagicMock(spec=anthropic.types.TextBlock)
    mock_text_block.text = text
    mock_message = MagicMock()
    mock_message.content = [mock_text_block]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_message
    return mock_client


def _make_pipeline(store: VectorStore) -> RAGPipeline:
    embedding_service = MagicMock(spec=EmbeddingService)
    embedding_service.get_embedding.side_effect = EmbeddingService.mock_embedding
    return RAGPipeline(
        store=store,
        embedding_service=embedding_service,
        anthropic_client=_make_client_mock(),
    )


def _make_pipeline_with_bad_response(store: VectorStore) -> RAGPipeline:
    """Pipeline whose LLM returns a non-TextBlock — triggers ValueError."""
    mock_message = MagicMock()
    mock_message.content = [MagicMock()]  # plain MagicMock ≠ TextBlock
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_message
    embedding_service = MagicMock(spec=EmbeddingService)
    embedding_service.get_embedding.side_effect = EmbeddingService.mock_embedding
    return RAGPipeline(
        store=store,
        embedding_service=embedding_service,
        anthropic_client=mock_client,
    )


def _make_pipeline_with_empty_content(store: VectorStore) -> RAGPipeline:
    """Pipeline whose LLM returns an empty content list — triggers ValueError."""
    mock_message = MagicMock()
    mock_message.content = []
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_message
    embedding_service = MagicMock(spec=EmbeddingService)
    embedding_service.get_embedding.side_effect = EmbeddingService.mock_embedding
    return RAGPipeline(
        store=store,
        embedding_service=embedding_service,
        anthropic_client=mock_client,
    )


# ── build_prompt ──────────────────────────────────────────────────────────────

def test_build_prompt_contains_question() -> None:
    pipeline = _make_pipeline(VectorStore())
    prompt = pipeline.build_prompt("What is the policy?", [])
    assert "What is the policy?" in prompt


def test_build_prompt_with_no_chunks_signals_missing_docs() -> None:
    pipeline = _make_pipeline(VectorStore())
    prompt = pipeline.build_prompt("Any question", [])
    assert "No relevant documents" in prompt


def test_build_prompt_injects_chunk_text() -> None:
    pipeline = _make_pipeline(VectorStore())
    chunks = [Chunk(text="Password must be 12 chars.", source="security.md", index=0)]
    prompt = pipeline.build_prompt("What is the password policy?", chunks)
    assert "Password must be 12 chars." in prompt


def test_build_prompt_includes_source() -> None:
    pipeline = _make_pipeline(VectorStore())
    chunks = [Chunk(text="Some text.", source="hr-policy.md", index=0)]
    prompt = pipeline.build_prompt("Question?", chunks)
    assert "hr-policy.md" in prompt


def test_build_prompt_numbers_multiple_chunks() -> None:
    pipeline = _make_pipeline(VectorStore())
    chunks = [
        Chunk(text="First chunk.", source="a.md", index=0),
        Chunk(text="Second chunk.", source="b.md", index=1),
    ]
    prompt = pipeline.build_prompt("Question?", chunks)
    assert "[1]" in prompt
    assert "[2]" in prompt


# ── query ─────────────────────────────────────────────────────────────────────

def test_query_returns_query_result() -> None:
    store = VectorStore()
    chunks = [Chunk(text="VPN required for remote work.", source="it.md", index=0)]
    embeddings = [EmbeddingService.mock_embedding(chunks[0].text)]
    store.add(chunks, embeddings)

    pipeline = _make_pipeline(store)
    result = pipeline.query("Do I need VPN?")

    assert isinstance(result, QueryResult)
    assert isinstance(result.answer, str)
    assert len(result.retrieved_chunks) > 0


def test_query_preview_truncated_to_limit() -> None:
    long_text = "word " * 200  # 200 words → well over 100 chars
    store = VectorStore()
    chunks = [Chunk(text=long_text.strip(), source="doc.md", index=0)]
    store.add(chunks, [EmbeddingService.mock_embedding(long_text)])

    pipeline = _make_pipeline(store)
    result = pipeline.query("anything")

    for meta in result.retrieved_chunks:
        assert len(meta.preview) <= CHUNK_PREVIEW_LENGTH


def test_query_with_empty_store_still_returns_result() -> None:
    pipeline = _make_pipeline(VectorStore())
    result = pipeline.query("What is the vacation policy?")
    assert isinstance(result.answer, str)
    assert result.retrieved_chunks == []


# ── summarize ─────────────────────────────────────────────────────────────────

def test_summarize_calls_llm_and_returns_string() -> None:
    pipeline = _make_pipeline(VectorStore())
    summary = pipeline.summarize("How many vacation days do I get per year?")
    assert isinstance(summary, str)
    assert len(summary) > 0


# ── error handling ─────────────────────────────────────────────────────────────

def test_summarize_raises_on_non_text_llm_response() -> None:
    """_call_llm raises ValueError when Claude returns a non-TextBlock."""
    pipeline = _make_pipeline_with_bad_response(VectorStore())
    with pytest.raises(ValueError, match="Unexpected non-text"):
        pipeline.summarize("any text")


def test_query_raises_on_non_text_llm_response() -> None:
    """Ensures the same guard applies in the query path."""
    pipeline = _make_pipeline_with_bad_response(VectorStore())
    with pytest.raises(ValueError, match="Unexpected non-text"):
        pipeline.query("What is the policy?")


def test_raises_on_empty_llm_content() -> None:
    """_call_llm raises ValueError when Claude returns empty content list."""
    pipeline = _make_pipeline_with_empty_content(VectorStore())
    with pytest.raises(ValueError, match="Empty response"):
        pipeline.summarize("any text")


def test_summarize_with_empty_input_still_calls_llm() -> None:
    """summarize() forwards even an empty string to the LLM without error."""
    pipeline = _make_pipeline(VectorStore())
    result = pipeline.summarize("")
    assert isinstance(result, str)


# ── stream_query ───────────────────────────────────────────────────────────────

def test_stream_query_first_event_is_chunks() -> None:
    store = VectorStore()
    chunks = [Chunk(text="Remote work requires VPN.", source="it.md", index=0)]
    store.add(chunks, [EmbeddingService.mock_embedding(chunks[0].text)])

    pipeline = _make_streaming_pipeline(store, tokens=["Answer"])
    events = list(pipeline.stream_query("VPN policy?"))

    assert events[0].startswith("event: chunks\n")


def test_stream_query_chunks_event_contains_valid_json() -> None:
    store = VectorStore()
    chunks = [Chunk(text="Password policy: 12 chars.", source="sec.md", index=0)]
    store.add(chunks, [EmbeddingService.mock_embedding(chunks[0].text)])

    pipeline = _make_streaming_pipeline(store, tokens=["ok"])
    events = list(pipeline.stream_query("password?"))

    data_line = events[0].split("\n")[1]
    payload = json.loads(data_line.removeprefix("data:").strip())
    assert isinstance(payload, list)
    assert payload[0]["source"] == "sec.md"


def test_stream_query_yields_token_events() -> None:
    store = VectorStore()
    chunks = [Chunk(text="Leave policy.", source="hr.md", index=0)]
    store.add(chunks, [EmbeddingService.mock_embedding(chunks[0].text)])

    pipeline = _make_streaming_pipeline(store, tokens=["Hello", " world"])
    events = list(pipeline.stream_query("leave?"))

    token_events = [e for e in events if e.startswith("event: token\n")]
    assert len(token_events) == 2

    def _token(event: str) -> str:
        return str(json.loads(event.split("\n")[1].removeprefix("data:").strip()))

    assert _token(token_events[0]) == "Hello"
    assert _token(token_events[1]) == " world"


def test_stream_query_last_event_is_done() -> None:
    store = VectorStore()
    chunks = [Chunk(text="Any content.", source="a.md", index=0)]
    store.add(chunks, [EmbeddingService.mock_embedding(chunks[0].text)])

    pipeline = _make_streaming_pipeline(store, tokens=["done"])
    events = list(pipeline.stream_query("question?"))

    assert events[-1] == "event: done\ndata: {}\n\n"


def test_stream_query_empty_store_yields_valid_sse() -> None:
    pipeline = _make_streaming_pipeline(VectorStore(), tokens=["answer"])
    events = list(pipeline.stream_query("anything?"))

    types = [e.split("\n")[0].removeprefix("event:").strip() for e in events]
    assert types[0] == "chunks"
    assert types[-1] == "done"
    assert all(e.endswith("\n\n") for e in events)
