import hashlib

from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
MOCK_SEED = 0xDEAD


class EmbeddingService:
    """Fetches embeddings from OpenAI or returns deterministic fakes for tests.

    Single Responsibility: only handles embedding API calls.
    Dependency Inversion: api_key injected via constructor.
    """

    def __init__(self, api_key: str) -> None:
        self._client = OpenAI(api_key=api_key)

    def get_embedding(self, text: str) -> list[float]:
        response = self._client.embeddings.create(
            input=text,
            model=EMBEDDING_MODEL,
        )
        return response.data[0].embedding

    @staticmethod
    def mock_embedding(text: str) -> list[float]:
        """Deterministic fake embedding — no API call, safe for tests."""
        digest = int(hashlib.sha256(text.encode()).hexdigest(), 16)
        seed = (digest ^ MOCK_SEED) & 0xFFFFFFFF
        values: list[float] = []
        for i in range(EMBEDDING_DIM):
            seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
            values.append((seed / 0xFFFFFFFF) * 2.0 - 1.0)
        magnitude = sum(v * v for v in values) ** 0.5
        return [v / magnitude for v in values]
