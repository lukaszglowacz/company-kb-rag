from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from main import app, get_pipeline
from rag.pipeline import ChunkMetadata, QueryResult, RAGPipeline


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def _mock_pipeline() -> MagicMock:
    pipeline = MagicMock(spec=RAGPipeline)
    return pipeline


# ── /health ───────────────────────────────────────────────────────────────────

def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# ── /query/stream ─────────────────────────────────────────────────────────────

def test_query_stream_returns_200_with_sse_content_type(
    client: TestClient,
) -> None:
    pipeline = _mock_pipeline()
    pipeline.stream_query.return_value = iter(
        ["event: done\ndata: {}\n\n"]
    )
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    try:
        response = client.post("/query/stream", json={"question": "test?"})
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
    finally:
        app.dependency_overrides.clear()


def test_query_stream_body_contains_chunks_and_done_events(
    client: TestClient,
) -> None:
    chunks_data = (
        '[{"source":"hr.md","score":0.9,"preview":"Holiday policy"}]'
    )
    pipeline = _mock_pipeline()
    pipeline.stream_query.return_value = iter([
        f"event: chunks\ndata: {chunks_data}\n\n",
        'event: token\ndata: "Holidays are 20 days."\n\n',
        "event: done\ndata: {}\n\n",
    ])
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    try:
        response = client.post("/query/stream", json={"question": "vacation?"})
        assert "event: chunks" in response.text
        assert "event: token" in response.text
        assert "event: done" in response.text
    finally:
        app.dependency_overrides.clear()


def test_query_stream_calls_pipeline_with_question(
    client: TestClient,
) -> None:
    pipeline = _mock_pipeline()
    pipeline.stream_query.return_value = iter(
        ["event: done\ndata: {}\n\n"]
    )
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    try:
        client.post("/query/stream", json={"question": "What is the leave policy?"})
        pipeline.stream_query.assert_called_once_with("What is the leave policy?")
    finally:
        app.dependency_overrides.clear()


def test_query_stream_empty_question_still_calls_pipeline(
    client: TestClient,
) -> None:
    pipeline = _mock_pipeline()
    pipeline.stream_query.return_value = iter(
        ["event: done\ndata: {}\n\n"]
    )
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    try:
        response = client.post("/query/stream", json={"question": ""})
        assert response.status_code == 200
        pipeline.stream_query.assert_called_once_with("")
    finally:
        app.dependency_overrides.clear()


# ── /query ────────────────────────────────────────────────────────────────────

def test_query_returns_answer_and_chunks(client: TestClient) -> None:
    pipeline = _mock_pipeline()
    pipeline.query.return_value = QueryResult(
        answer="VPN is required.",
        retrieved_chunks=[
            ChunkMetadata(source="it.md", score=0.95, preview="VPN required"),
        ],
    )
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    try:
        response = client.post("/query", json={"question": "VPN?"})
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "VPN is required."
        assert data["retrieved_chunks"][0]["source"] == "it.md"
    finally:
        app.dependency_overrides.clear()
