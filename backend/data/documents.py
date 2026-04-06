import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent


def load_documents() -> list[tuple[str, str]]:
    """Return (text, source) pairs for every .md file in the data directory."""
    docs = []
    for path in sorted(DATA_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        docs.append((text, path.name))
        logger.info("Loaded document: %s (%d chars)", path.name, len(text))
    return docs
