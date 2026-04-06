from dataclasses import dataclass


CHUNK_SIZE_WORDS = 100
OVERLAP_WORDS = 50


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    index: int


class TextChunker:
    """Splits text into overlapping word-based chunks.

    Single Responsibility: only handles text splitting logic.
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE_WORDS,
        overlap: int = OVERLAP_WORDS,
    ) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self._chunk_size = chunk_size
        self._overlap = overlap

    def chunk(self, text: str, source: str) -> list[Chunk]:
        words = text.split()
        if not words:
            return []

        chunks: list[Chunk] = []
        step = self._chunk_size - self._overlap
        index = 0

        for start in range(0, len(words), step):
            window = words[start : start + self._chunk_size]
            chunks.append(Chunk(text=" ".join(window), source=source, index=index))
            index += 1
            if start + self._chunk_size >= len(words):
                break

        return chunks
