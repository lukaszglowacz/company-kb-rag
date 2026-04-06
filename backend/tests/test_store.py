from rag.chunker import Chunk
from rag.embeddings import EmbeddingService
from rag.store import VectorStore


def _make_chunk(text: str, index: int = 0) -> Chunk:
    return Chunk(text=text, source="test", index=index)


def test_empty_store_returns_no_results() -> None:
    store = VectorStore()
    query = EmbeddingService.mock_embedding("anything")
    assert store.search(query, top_k=5) == []


def test_add_increases_count() -> None:
    store = VectorStore()
    chunks = [_make_chunk("hello world")]
    embeddings = [EmbeddingService.mock_embedding("hello world")]
    store.add(chunks, embeddings)
    assert store.count == 1


def test_search_returns_top_k() -> None:
    store = VectorStore()
    texts = ["alpha", "beta", "gamma", "delta", "epsilon"]
    chunks = [_make_chunk(t, i) for i, t in enumerate(texts)]
    embeddings = [EmbeddingService.mock_embedding(t) for t in texts]
    store.add(chunks, embeddings)

    results = store.search(EmbeddingService.mock_embedding("alpha"), top_k=3)
    assert len(results) == 3


def test_search_top_k_larger_than_store() -> None:
    store = VectorStore()
    chunks = [_make_chunk("only one")]
    embeddings = [EmbeddingService.mock_embedding("only one")]
    store.add(chunks, embeddings)

    results = store.search(EmbeddingService.mock_embedding("only one"), top_k=10)
    assert len(results) == 1


def test_exact_match_scores_highest() -> None:
    store = VectorStore()
    texts = ["the quick brown fox", "lazy dog", "jumped over"]
    chunks = [_make_chunk(t, i) for i, t in enumerate(texts)]
    embeddings = [EmbeddingService.mock_embedding(t) for t in texts]
    store.add(chunks, embeddings)

    query = EmbeddingService.mock_embedding("the quick brown fox")
    results = store.search(query, top_k=3)

    top_chunk, top_score = results[0]
    assert top_chunk.text == "the quick brown fox"
    assert top_score > 0.99  # cosine similarity of identical vectors ≈ 1.0


def test_results_ordered_by_descending_score() -> None:
    store = VectorStore()
    texts = ["cat", "dog", "fish", "bird"]
    chunks = [_make_chunk(t, i) for i, t in enumerate(texts)]
    embeddings = [EmbeddingService.mock_embedding(t) for t in texts]
    store.add(chunks, embeddings)

    results = store.search(EmbeddingService.mock_embedding("cat"), top_k=4)
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_scores_are_floats() -> None:
    store = VectorStore()
    chunks = [_make_chunk("hello")]
    embeddings = [EmbeddingService.mock_embedding("hello")]
    store.add(chunks, embeddings)

    results = store.search(EmbeddingService.mock_embedding("hello"), top_k=1)
    assert isinstance(results[0][1], float)
