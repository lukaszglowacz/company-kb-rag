import pytest

from rag.chunker import TextChunker


def test_empty_text_returns_no_chunks() -> None:
    chunker = TextChunker(chunk_size=10, overlap=3)
    assert chunker.chunk("", "src") == []


def test_whitespace_only_returns_no_chunks() -> None:
    chunker = TextChunker(chunk_size=10, overlap=3)
    assert chunker.chunk("   \n\t  ", "src") == []


def test_single_word_returns_one_chunk() -> None:
    chunker = TextChunker(chunk_size=10, overlap=3)
    result = chunker.chunk("hello", "src")
    assert len(result) == 1
    assert result[0].text == "hello"
    assert result[0].source == "src"
    assert result[0].index == 0


def test_normal_text_chunk_count() -> None:
    # 10 words, size=5, overlap=2 → step=3 → starts: 0, 3, 6 → 3 chunks
    words = " ".join(f"w{i}" for i in range(10))
    chunker = TextChunker(chunk_size=5, overlap=2)
    result = chunker.chunk(words, "doc")
    assert len(result) == 3


def test_chunk_source_preserved() -> None:
    chunker = TextChunker(chunk_size=5, overlap=2)
    words = " ".join(f"w{i}" for i in range(10))
    result = chunker.chunk(words, "my-doc")
    assert all(c.source == "my-doc" for c in result)


def test_overlap_correctness() -> None:
    # words: a b c d e f g  (7 words), size=4, overlap=2 → step=2
    # chunk 0: a b c d
    # chunk 1: c d e f   (overlap: c d)
    # chunk 2: e f g     (last window)
    chunker = TextChunker(chunk_size=4, overlap=2)
    result = chunker.chunk("a b c d e f g", "src")
    assert result[0].text == "a b c d"
    assert result[1].text == "c d e f"
    assert "c" in result[1].text  # overlap word present
    assert "d" in result[1].text  # overlap word present


def test_chunk_indices_sequential() -> None:
    chunker = TextChunker(chunk_size=5, overlap=2)
    words = " ".join(f"w{i}" for i in range(20))
    result = chunker.chunk(words, "src")
    assert [c.index for c in result] == list(range(len(result)))


def test_invalid_overlap_raises() -> None:
    with pytest.raises(ValueError, match="overlap must be smaller than chunk_size"):
        TextChunker(chunk_size=5, overlap=5)


def test_chunk_is_immutable() -> None:
    chunker = TextChunker(chunk_size=5, overlap=2)
    result = chunker.chunk("a b c d e", "src")
    with pytest.raises(Exception):
        result[0].text = "changed"  # type: ignore[misc]
