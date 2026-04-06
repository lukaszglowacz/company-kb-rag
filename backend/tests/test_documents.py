from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from data.documents import load_documents

_CHUNK_WORD_REPEAT = 50   # enough words to produce at least one chunk
_EMBEDDING_DIM = 128      # dummy embedding dimension for mocks


# ── load_documents ────────────────────────────────────────────────────────────

def test_load_documents_returns_text_and_source(tmp_path: Path) -> None:
    (tmp_path / "guide.md").write_text("Hello world", encoding="utf-8")
    with patch("data.documents.DATA_DIR", tmp_path):
        docs = load_documents()
    assert len(docs) == 1
    text, source = docs[0]
    assert text == "Hello world"
    assert source == "guide.md"


def test_load_documents_ignores_non_md_files(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")
    (tmp_path / "doc.md").write_text("keep me", encoding="utf-8")
    with patch("data.documents.DATA_DIR", tmp_path):
        docs = load_documents()
    assert len(docs) == 1
    assert docs[0][1] == "doc.md"


def test_load_documents_returns_empty_list_when_no_md_files(
    tmp_path: Path,
) -> None:
    with patch("data.documents.DATA_DIR", tmp_path):
        docs = load_documents()
    assert docs == []


def test_load_documents_returns_sorted_by_filename(tmp_path: Path) -> None:
    (tmp_path / "z_last.md").write_text("Z", encoding="utf-8")
    (tmp_path / "a_first.md").write_text("A", encoding="utf-8")
    with patch("data.documents.DATA_DIR", tmp_path):
        docs = load_documents()
    assert [source for _, source in docs] == ["a_first.md", "z_last.md"]


# ── _ingest_documents ─────────────────────────────────────────────────────────

def test_ingest_documents_calls_embedding_service_for_each_chunk(
    tmp_path: Path,
) -> None:
    (tmp_path / "doc.md").write_text("A " * _CHUNK_WORD_REPEAT, encoding="utf-8")

    mock_embedding = MagicMock()
    mock_embedding.get_embedding.return_value = [0.1] * _EMBEDDING_DIM

    with (
        patch("data.documents.DATA_DIR", tmp_path),
        patch("main._store") as mock_store,
        patch("main._chunker") as mock_chunker,
    ):
        from rag.chunker import Chunk

        fake_chunk = Chunk(text="A " * 50, source="doc.md", index=0)
        mock_chunker.chunk.return_value = [fake_chunk]
        mock_store.count = 1

        from main import _ingest_documents

        _ingest_documents(mock_embedding)

    mock_embedding.get_embedding.assert_called_once_with(fake_chunk.text)
    mock_store.add.assert_called_once()


def test_ingest_documents_skips_empty_data_dir(tmp_path: Path) -> None:
    mock_embedding = MagicMock()

    with (
        patch("data.documents.DATA_DIR", tmp_path),
        patch("main._store") as mock_store,
    ):
        mock_store.count = 0
        from main import _ingest_documents

        _ingest_documents(mock_embedding)

    mock_embedding.get_embedding.assert_not_called()
    mock_store.add.assert_not_called()


# ── _get_openai_api_key ───────────────────────────────────────────────────────

def test_get_openai_api_key_returns_env_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    from main import _get_openai_api_key

    assert _get_openai_api_key() == "sk-test-123"


def test_get_openai_api_key_returns_none_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from main import _get_openai_api_key

    assert _get_openai_api_key() is None
