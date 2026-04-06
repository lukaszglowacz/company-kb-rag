from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from rag.chunker import Chunk


@dataclass(frozen=True)
class StoredChunk:
    chunk: Chunk
    embedding: npt.NDArray[np.float32]


class VectorStore:
    """In-memory vector store with cosine similarity search.

    Single Responsibility: only handles storage and retrieval.
    No side effects on import.
    """

    def __init__(self) -> None:
        self._items: list[StoredChunk] = []

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        for chunk, embedding in zip(chunks, embeddings):
            vector = np.array(embedding, dtype=np.float32)
            self._items.append(StoredChunk(chunk=chunk, embedding=vector))

    def search(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[tuple[Chunk, float]]:
        if not self._items:
            return []

        query = np.array(query_embedding, dtype=np.float32)
        query = query / (np.linalg.norm(query) + 1e-10)

        matrix = np.stack([item.embedding for item in self._items])
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
        matrix = matrix / norms

        scores: npt.NDArray[np.float32] = matrix @ query
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            (self._items[i].chunk, float(scores[i]))
            for i in top_indices
        ]

    @property
    def count(self) -> int:
        return len(self._items)
