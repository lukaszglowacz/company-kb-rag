import hashlib

from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
MOCK_SEED = 0xDEAD       # Arbitrary seed for deterministic test embeddings
LCG_MULTIPLIER = 1664525      # Linear Congruential Generator multiplier (Knuth)
LCG_INCREMENT = 1013904223    # Linear Congruential Generator increment
UINT32_MASK = 0xFFFFFFFF      # 32-bit unsigned integer mask


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
        seed = (digest ^ MOCK_SEED) & UINT32_MASK
        values: list[float] = []
        for _ in range(EMBEDDING_DIM):
            seed = (seed * LCG_MULTIPLIER + LCG_INCREMENT) & UINT32_MASK
            values.append((seed / UINT32_MASK) * 2.0 - 1.0)
        magnitude = sum(v * v for v in values) ** 0.5
        return [v / magnitude for v in values]
