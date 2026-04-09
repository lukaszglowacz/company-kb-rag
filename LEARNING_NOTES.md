## 1. Project Overview

### What was built

**Company KB RAG** is a company knowledge base chatbot based on the
Retrieval-Augmented Generation (RAG) architecture. The user asks a question in natural language
(in Polish or English), the system semantically searches for similar fragments from company
documents (HR, onboarding, tech stack, company overview), and then Claude (Anthropic)
generates an answer based solely on the retrieved fragments — without hallucinations.
The application consists of a FastAPI backend (Python) handling the RAG pipeline with SSE streaming
and a Next.js 14 frontend with a real-time chat interface.

### Final architecture

```
User
  │
  │  POST /query/stream  (text/event-stream)
  ▼
┌─────────────────────────────────────────────────────┐
│  Next.js 14 App Router  (port 3000)                 │
│                                                     │
│  page.tsx → ChatWindow                              │
│               ├── ChatMessages → MessageBubble      │
│               │                  └── useTypewriter  │
│               ├── ChatInput                         │
│               ├── PromptHistory                     │
│               └── ChunksSidebar                     │
│                                                     │
│  hooks/useChat.ts                                   │
│  lib/api.ts  →  queryStream() → fetch SSE           │
└───────────────────────┬─────────────────────────────┘
                        │  fetch  (text/event-stream)
                        ▼
┌─────────────────────────────────────────────────────┐
│  FastAPI  (port 8000)                               │
│                                                     │
│  main.py                                            │
│    ├── lifespan()  ← auto-ingestion on startup      │
│    ├── POST /ingest                                 │
│    ├── POST /query                                  │
│    ├── POST /query/stream  ← SSE                    │
│    ├── POST /summarize                              │
│    └── GET  /health                                 │
│                                                     │
│  RAGPipeline (rag/pipeline.py)                      │
│    ├── _retrieve()                                  │
│    │     ├── EmbeddingService.get_embedding()       │
│    │     └── VectorStore.search()                   │
│    ├── query()                                      │
│    ├── stream_query()                               │
│    └── summarize()                                  │
│                                                     │
│  TextChunker      (rag/chunker.py)                  │
│  EmbeddingService (rag/embeddings.py) → OpenAI API  │
│  VectorStore      (rag/store.py)      → NumPy       │
│  load_documents   (data/documents.py) → *.md        │
└───────────────────────┬─────────────────────────────┘
                        │  messages.stream()
                        ▼
              Anthropic Claude API
              (claude-sonnet-4-20250514)

┌─────────────────────────────────────────────────────┐
│  Docker Compose                                     │
│  backend (python:3.11-slim) + frontend (node:20)    │
│  Internal network: kb-network                       │
└─────────────────────────────────────────────────────┘
```

### Technology stack

| Component | Technology | Rationale |
|---|---|---|
| Backend framework | FastAPI (Python 3.11) | Async, automatic OpenAPI, Pydantic, `Depends()` DI, `StreamingResponse` SSE |
| LLM | Anthropic Claude (claude-sonnet-4) | Best response quality, native SSE streaming support in SDK |
| Embeddings | OpenAI text-embedding-3-small | Multilingual, 1536 dim, best price-to-quality ratio |
| Vector store | NumPy (in-memory) | Zero external dependencies, sufficient for small knowledge base, educationally transparent |
| Frontend | Next.js 14 App Router | SSR, `output: "standalone"` for Docker, `"use client"` granularity |
| Styling | Tailwind CSS 3 | Utility-first, no separate CSS files, responsiveness in one place |
| Containerization | Docker Compose | Single-command startup of the entire stack, network isolation |
| CI | GitHub Actions | 7 parallel jobs: flake8, mypy, pytest, tsc, ESLint, Vitest, next build |
| Backend typing | mypy --strict | Catches type errors in CI, eliminates `Any` |
| Frontend typing | TypeScript strict | All API responses typed via `frontend/types/index.ts` |
| Backend tests | pytest + pytest-cov | Python standard; `mock_embedding()` allows offline testing |
| Frontend tests | Vitest | Native ESM, faster than Jest for TypeScript |

---

## 2. RAG Concepts — What I Learned

### What is an LLM and how do we communicate with it via API

**What it is:** A Large Language Model is a language model trained on massive text datasets,
capable of generating coherent responses in natural language.

**How communication works:**

```python
# backend/rag/pipeline.py — _call_llm()
message = self._client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system=SYSTEM_PROMPT,          # system instructions
    messages=[{"role": "user", "content": prompt}],  # question + context
)
return message.content[0].text     # text response
```

In streaming mode:

```python
# backend/rag/pipeline.py — stream_query()
with self._client.messages.stream(
    model=CLAUDE_MODEL,
    max_tokens=MAX_TOKENS_ANSWER,
    system=SYSTEM_PROMPT,
    messages=[{"role": "user", "content": prompt}],
) as stream:
    for text in stream.text_stream:  # token iterator
        yield f"event: token\ndata: {json.dumps(text)}\n\n"
```

**Where in code:** `backend/rag/pipeline.py` — methods `_call_llm()` and `stream_query()`.

---

### What are embeddings and why they matter

**What it is:** An embedding is a representation of text as a vector of floating-point numbers
(here: 1536 float32 numbers). Semantically similar texts have geometrically close vectors.

**Implementation:**

```python
# backend/rag/embeddings.py
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

def get_embedding(self, text: str) -> list[float]:
    response = self._client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL,
    )
    return response.data[0].embedding  # list of 1536 numbers
```

**Why they matter:** Allows comparing a user's question with document fragments
without exact keyword matching. The question "how many vacation days?" and the text
"26 days annual leave" have high similarity in embedding space.

> **Key observation:** `text-embedding-3-small` is multilingual, but a cross-lingual
> semantic gap exists — "komputer" (PL) vs "laptop" (EN) may have lower similarity
> than expected.

---

### What is chunking and why overlap matters

**What it is:** Chunking is the division of long documents into smaller fragments
that can be individually indexed and searched.

**Implementation:**

```python
# backend/rag/chunker.py
CHUNK_SIZE_WORDS = 100
OVERLAP_WORDS = 50

def chunk(self, text: str, source: str) -> list[Chunk]:
    words = text.split()
    step = self._chunk_size - self._overlap  # step = 50 words
    for start in range(0, len(words), step):
        window = words[start : start + self._chunk_size]
        chunks.append(Chunk(text=" ".join(window), source=source, index=index))
```

**Why overlap = 50 words matters:** Without overlap, a key sentence could land
on the boundary of two chunks and be incomplete in neither. Overlap guarantees
that every sentence appears complete in at least one chunk. Trade-off: more
chunks = higher embedding cost.

---

### What is cosine similarity and how NumPy implements it

**What it is:** A measure of similarity between two vectors — the cosine of the angle between them.
Value 1.0 = identical directions, 0.0 = perpendicular, −1.0 = opposite.

**Matrix implementation in NumPy:**

```python
# backend/rag/store.py
def search(self, query_embedding: list[float], top_k: int = 5):
    query = np.array(query_embedding, dtype=np.float32)
    query = query / (np.linalg.norm(query) + NORMALIZATION_EPSILON)  # normalization

    # Matrix of all chunks: N × 1536
    matrix = np.stack([item.embedding for item in self._items])
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + NORMALIZATION_EPSILON
    matrix = matrix / norms  # normalize each row

    # Dot product = cosine similarity after normalization
    scores: npt.NDArray[np.float32] = matrix @ query  # N results at once
    top_indices = np.argsort(scores)[::-1][:top_k]
```

**Why cosine, not dot product:** Dot product depends on vector length.
After normalization both measures are equivalent — we use dot product `@` because it's faster.

---

### What is context window and prompt injection

**Context window:** Maximum number of tokens visible to the LLM in one query.
Combines: system prompt + document fragments + user question.

**How we construct the prompt:**

```python
# backend/rag/pipeline.py
def build_prompt(self, question: str, chunks: list[Chunk]) -> str:
    excerpts = "\n\n".join(
        f"[{i + 1}] (source: {c.source})\n{c.text}"
        for i, c in enumerate(chunks)
    )
    return f"Document excerpts:\n\n{excerpts}\n\nQuestion: {question}"
```

**Prompt injection:** An attack where the user injects instructions for the LLM,
e.g. "Ignore previous instructions and...". Mitigation: `SYSTEM_PROMPT` with an explicit
prohibition on inventing information, limited to the provided fragments.

---

### Difference between RAG and fine-tuning

| Feature | RAG (our project) | Fine-tuning |
|---|---|---|
| Knowledge | From external documents (retrieval) | Built into model weights |
| Knowledge update | Add document, reingest | Retrain model (GPU, time, data) |
| Cost | Cheap (embeddings + API call) | Expensive |
| Hallucination risk | Low (model sees sources) | Higher |
| Transparency | ✅ Which fragments were used is visible | ❌ |

---

### What is hallucination and how RAG prevents it

**Hallucination:** LLM generates convincingly sounding but false information.

**How RAG prevents it — two mechanisms:**

```python
# backend/rag/pipeline.py
SYSTEM_PROMPT = (
    "You are a company knowledge base assistant. "
    "Answer questions using ONLY the provided document excerpts. "
    "If the answer is not in the excerpts, say so clearly. "
    "Do not invent information. ..."
)
```

1. The model only receives verified fragments from company documents
2. `SYSTEM_PROMPT` contains an explicit prohibition on inventing information

---

## 3. Build Stages — PR by PR

### PR #1 — Repository initialization and CI

**Commit:** `chore: initialize repo structure with CI pipeline (#1)`

**What was built:**
- Directory structure: `backend/`, `frontend/`, `.github/workflows/`
- CI pipeline with 7 parallel jobs
- `requirements.txt`, base configuration files

**CI workflow (`.github/workflows/ci.yml`):**

```yaml
jobs:
  backend-lint:
    name: Backend — flake8
    steps:
      - run: flake8 . --max-line-length=88 --exclude=__pycache__

  backend-typecheck:
    name: Backend — mypy --strict
    steps:
      - run: mypy . --strict --ignore-missing-imports

  backend-test:
    name: Backend — pytest + coverage
    steps:
      - run: pytest tests/ --cov=. --cov-report=term-missing

  frontend-typecheck:
    name: Frontend — tsc --noEmit
  frontend-lint:
    name: Frontend — ESLint
  frontend-test:
    name: Frontend — Vitest
  frontend-build:
    name: Frontend — next build
```

**Key decisions:** `mypy --strict` from the start — enforces full typing,
eliminates errors that would otherwise surface in production.

---

### PR #2 — Ingestion pipeline

**Commit:** `feat: implement ingestion pipeline — TextChunker, VectorStore, EmbeddingService (#2)`

**What was built:**
- `TextChunker` — text splitting into overlapping chunks
- `VectorStore` — in-memory storage with cosine similarity (NumPy)
- `EmbeddingService` — OpenAI API wrapper + `mock_embedding()` for tests

**SRP principle — each class has one responsibility:**

```python
# backend/rag/chunker.py — ONLY text splitting
class TextChunker:
    def chunk(self, text: str, source: str) -> list[Chunk]: ...

# backend/rag/store.py — ONLY storage and search
class VectorStore:
    def add(self, chunks, embeddings): ...
    def search(self, query_embedding, top_k): ...

# backend/rag/embeddings.py — ONLY OpenAI communication
class EmbeddingService:
    def get_embedding(self, text: str) -> list[float]: ...
```

**Mock embedding — tests without API call:**

```python
# backend/rag/embeddings.py
@staticmethod
def mock_embedding(text: str) -> list[float]:
    """Deterministic fake — same text always gives same vector."""
    digest = int(hashlib.sha256(text.encode()).hexdigest(), 16)
    seed = (digest ^ MOCK_SEED) & UINT32_MASK
    # LCG pseudorandom generator
    values: list[float] = []
    for _ in range(EMBEDDING_DIM):
        seed = (seed * LCG_MULTIPLIER + LCG_INCREMENT) & UINT32_MASK
        values.append((seed / UINT32_MASK) * 2.0 - 1.0)
    magnitude = sum(v * v for v in values) ** 0.5
    return [v / magnitude for v in values]
```

---

### PR #3 — RAG Query Pipeline

**Commit:** `feat: implement RAG query pipeline, summarization endpoint, health check (#3)`

**What was built:**
- `RAGPipeline` — retrieval + LLM orchestration
- Endpoints: `POST /query`, `POST /summarize`, `GET /health`

**DIP principle — dependencies injected via constructor:**

```python
# backend/rag/pipeline.py
class RAGPipeline:
    def __init__(
        self,
        store: VectorStore,                    # injected
        embedding_service: EmbeddingService,   # injected
        anthropic_client: anthropic.Anthropic, # injected
    ) -> None:
        self._store = store
        self._embedding_service = embedding_service
        self._client = anthropic_client
```

RAGPipeline does not create its own dependencies — it receives them from outside.
In tests, MagicMock can be injected instead of the real client.

---

### PR #4 — Next.js 14 Chat UI

**Commit:** `feat: implement Next.js 14 chat UI for RAG knowledge base (#4)`

**What was built:**
- Components: `ChatWindow`, `ChatMessages`, `MessageBubble`, `ChatInput`, `ChunksSidebar`
- Hook `useChat` — message state management
- `lib/api.ts` — SSE client
- `types/index.ts` — TypeScript interfaces

**TypeScript types (`frontend/types/index.ts`):**

```typescript
export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  chunks?: ChunkMetadata[];
  isStreaming?: boolean;
}

export interface StreamCallbacks {
  onChunks: (chunks: ChunkMetadata[]) => void;
  onToken: (token: string) => void;
  onDone: () => void;
  onError: (message: string) => void;
}
```

**SSE parser (`frontend/lib/api.ts`):**

```typescript
const reader = res.body.getReader();
const decoder = new TextDecoder();
let buffer = "";

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  buffer += decoder.decode(value, { stream: true });

  const events = buffer.split("\n\n");  // SSE events separated by \n\n
  buffer = events.pop() ?? "";           // last incomplete one — stays in buffer

  for (const event of events) {
    const type = event.split("\n").find(l => l.startsWith("event:"))
      ?.slice("event:".length).trim();
    const data = event.split("\n").find(l => l.startsWith("data:"))
      ?.slice("data:".length).trim();
    if (type && data) dispatchSSEEvent(type, data, callbacks);
  }
}
```

---

### PR #5 — Company documents and auto-ingestion

**Commit:** `feat(backend): company documents + auto-ingestion on startup (#5)`

**What was built:**
- 4 Markdown documents: `hr_policy.md`, `onboarding.md`, `tech_stack.md`, `company_overview.md`
- `data/documents.py` — loading all `.md` files
- `lifespan` — auto-ingestion on startup

**lifespan context manager:**

```python
# backend/main.py
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    api_key = _get_openai_api_key()
    if api_key:
        try:
            _ingest_documents(EmbeddingService(api_key=api_key))
        except Exception:
            logger.exception("Auto-ingestion failed — continuing without pre-loaded docs")
    else:
        logger.warning("OPENAI_API_KEY not set — skipping auto-ingestion")
    yield  # application runs and handles requests
    # cleanup after yield (on shutdown)

app = FastAPI(title="Company KB RAG API", lifespan=lifespan)
```

**Dependency Injection in FastAPI (`Depends()`):**

```python
# backend/main.py
def get_pipeline(
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    anthropic_client: anthropic.Anthropic = Depends(get_anthropic_client),
) -> RAGPipeline:
    return RAGPipeline(store=_store, embedding_service=embedding_service,
                       anthropic_client=anthropic_client)

@app.post("/query/stream")
def query_stream(
    request: QueryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),  # FastAPI injects automatically
) -> StreamingResponse:
    return StreamingResponse(pipeline.stream_query(request.question),
                             media_type="text/event-stream")
```

---

### PR #6 — Streaming SSE

**Commit:** `feat: streaming SSE query endpoint + streaming chat UI (#6)`

**What was built:**
- `POST /query/stream` with `StreamingResponse`
- `stream_query()` — SSE event generator
- Frontend: `useChat` with token handling

**SSE format — three event types:**

```
event: chunks
data: [{"source":"hr_policy.md","score":0.55,"preview":"26 days annual..."}]

event: token
data: "Based on"

event: token
data: " the HR documents..."

event: done
data: {}
```

**SSE generator (`backend/rag/pipeline.py`):**

```python
def stream_query(self, question: str) -> Iterator[str]:
    metadata, prompt = self._retrieve(question)  # shared logic with query()

    # Event 1: found fragments (immediately, before generation)
    chunks_payload = json.dumps(
        [{"source": m.source, "score": m.score, "preview": m.preview}
         for m in metadata]
    )
    yield f"event: chunks\ndata: {chunks_payload}\n\n"

    # Events 2-N: tokens from Claude (streaming)
    with self._client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS_ANSWER,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            yield f"event: token\ndata: {json.dumps(text)}\n\n"

    # Final event
    yield "event: done\ndata: {}\n\n"
```

**DRY elimination — `_retrieve()` helper:**

```python
def _retrieve(self, question: str) -> tuple[list[ChunkMetadata], str]:
    """Shared logic for query() and stream_query() — DRY."""
    query_embedding = self._embedding_service.get_embedding(question)
    results = self._store.search(query_embedding, top_k=TOP_K_CHUNKS)
    chunks = [chunk for chunk, _ in results]
    metadata = [
        ChunkMetadata(source=c.source, score=s,
                      preview=(c.text or "")[:CHUNK_PREVIEW_LENGTH])
        for c, s in results
    ]
    prompt = self.build_prompt(question, chunks)
    return metadata, prompt
```

---

### PR #7 — Docker Compose

**Commit:** `Feat/etap 7 docker (#7)`

**What was built:**
- `docker-compose.yml` with two services and shared network
- `backend/Dockerfile` — python:3.11-slim, non-root user
- `frontend/Dockerfile` — 3-stage Node.js standalone build
- `frontend/next.config.mjs` with `output: "standalone"`

**`docker-compose.yml`:**

```yaml
services:
  backend:
    build: ./backend
    ports: ["8000:8000"]
    env_file: .env
    networks: [kb-network]
    restart: unless-stopped

  frontend:
    build:
      context: ./frontend
      args:
        NEXT_PUBLIC_API_URL: ${NEXT_PUBLIC_API_URL:-http://localhost:8000}
    ports: ["3000:3000"]
    depends_on: [backend]
    networks: [kb-network]
    restart: unless-stopped

networks:
  kb-network:
    driver: bridge
```

**3-stage frontend Dockerfile:**

```dockerfile
# Stage 1: dependency installation
FROM node:20-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci

# Stage 2: application build
FROM node:20-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

# Stage 3: minimal runtime (only what's needed)
FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production
RUN addgroup --system --gid 1001 nodejs \
 && adduser --system --uid 1001 nextjs
COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static
USER nextjs           # non-root for security
EXPOSE 3000
CMD ["node", "server.js"]
```

---

### PR #8 — UI fixes, quality, responsiveness

**Commit:** `Fix/markdown rendering and language (#8)`

**What was built:**
- `react-markdown` in `MessageBubble` — rendering headings, lists, bold text
- `useTypewriter` hook — smooth typing effect
- Responsive `ChunksSidebar` — mobile bottom sheet + desktop sidebar
- Strengthened `SYSTEM_PROMPT` — enforcing response language
- `TOP_K_CHUNKS` increased from 5 to 10
- `postcss.config.js`, `public/.gitkeep`, `CORSMiddleware`, `*.tsbuildinfo` in `.gitignore`

**Hook `useTypewriter` (`frontend/hooks/useTypewriter.ts`):**

```typescript
const CHARS_PER_TICK = 4;
const TICK_MS = 18;

export function useTypewriter(text: string, active: boolean): string {
  const [pos, setPos] = useState(0);
  const textRef = useRef(text);
  textRef.current = text;  // ref = always current value without recreating interval

  useEffect(() => {
    if (!active) return;
    const id = setInterval(() => {
      setPos((p) => {
        const target = textRef.current.length;
        return p + CHARS_PER_TICK >= target ? target : p + CHARS_PER_TICK;
      });
    }, TICK_MS);
    return () => clearInterval(id);
  }, [active]);  // one interval for the entire streaming

  if (!active) return text;   // old messages — full text immediately
  return text.slice(0, pos);  // streaming — characters one by one
}
```

---

## 4. Bugs and Fixes

### Bug #1 — flake8 E501: line too long

**What it was:** Line 24 in `rag/pipeline.py` had 106 characters, flake8 limit is 88.

**Before:**
```python
"If the user writes in Polish, respond 100% in Polish — even if the source documents are in English. "
```

**After:**
```python
"If the user writes in Polish, respond 100% in Polish — "
"even if the source documents are in English. "
```

**Lesson:** Run `flake8 . --max-line-length=88` locally before every commit.

---

### Bug #2 — CORS error (browser was blocking requests)

**What it was:** Browser was blocking fetch from `localhost:3000` to `localhost:8000`.

**Cause:** Same-Origin Policy — browser requires
`Access-Control-Allow-Origin` header for cross-origin requests.

**Fix (`backend/main.py`):**
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Lesson:** Always configure CORS when frontend and backend are on different ports.

---

### Bug #3 — Missing postcss.config.js

**What it was:** Tailwind CSS was not loading styles in Docker — the entire UI looked like unstyled HTML.

**Cause:** Next.js requires `postcss.config.js` to process `@tailwind` directives.
The file was on one branch and didn't make it to `main`.

**Fix:**
```javascript
// frontend/postcss.config.js
module.exports = {
  plugins: { tailwindcss: {}, autoprefixer: {} },
};
```

**Lesson:** Framework configuration files must be committed.

---

### Bug #4 — Missing frontend/public/

**What it was:** Docker build failed: `"/app/public": not found`.

**Cause:** Dockerfile copies `public/` directory in the runner stage,
but the directory did not exist in the repository.

**Fix:** Created `frontend/public/.gitkeep` — empty file forcing
git to track the directory.

**Lesson:** Next.js standalone build requires `public/` directory even if empty.

---

### Bug #5 — nginx in frontend Dockerfile

**What it was:** Frontend was not working in Docker — nginx cannot serve Next.js SSR.

**Cause:** nginx serves static files, but Next.js requires Node.js runtime.

**Before:**
```dockerfile
FROM nginx:alpine
COPY --from=builder /app/out /usr/share/nginx/html
```

**After:**
```dockerfile
FROM node:20-alpine AS runner
CMD ["node", "server.js"]
```

**Lesson:** Next.js with `output: "standalone"` generates its own Node.js server.

---

### Bug #6 — `import json` inside a function

**What it was:** `import json` was inside the `stream_query()` method instead of at the top of the file.

**Fix:** Moved to top-level (line 1 of file).

**Lesson:** Imports always at the top of the file — isort/flake8 requirement.

---

### Bug #7 — `anthropic.omit` ImportError on CI

**Cause:** `omit` sentinel does not exist in the older Anthropic SDK version on CI.

**Before:**
```python
from anthropic import omit as OMIT
kwargs["system"] = system if system is not None else OMIT
```

**After:**
```python
kwargs: dict[str, object] = {
    "model": CLAUDE_MODEL,
    "max_tokens": max_tokens,
    "messages": [{"role": "user", "content": prompt}],
}
if system is not None:
    kwargs["system"] = system
message = self._client.messages.create(**kwargs)  # type: ignore[call-overload]
```

**Lesson:** Do not use SDK features that may not exist in older versions.

---

### Bug #8 — `@pytest.fixture()` untyped decorator

**What it was:** mypy --strict was failing on CI: `error: Untyped decorator makes function untyped`.

**Cause:** Older pytest stubs on CI did not have types for `@pytest.fixture()`.

**Before:**
```python
@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)
```

**After:**
```python
def _make_client() -> TestClient:
    return TestClient(app)
```

**Lesson:** With `mypy --strict`, avoid decorators without full types in older libraries.

---

## 5. Code Review Notes

### Note #1 — SRP violation: `_ingest_documents` was doing too much

**Problem:** The function was doing: load → chunk → embed → store — four responsibilities.

**Solution:** Split into separate calls:
```python
docs = load_documents()
for text, source in docs:
    chunks = _chunker.chunk(text, source)
    embeddings = [embedding_service.get_embedding(c.text) for c in chunks]
    _store.add(chunks, embeddings)
```

**Lesson:** If a function description contains "and", split it.

---

### Note #2 — DRY violation: duplicated retrieve logic

**Problem:** `query()` and `stream_query()` repeated identical embed + search logic.

**Solution:** `_retrieve()` helper — one definition, two calls.

**Lesson:** DRY — every logic in one place.

---

### Note #3 — Magic numbers

**Problem:** Numbers `5`, `100`, `1024` without explanation in code.

**Solution:**
```python
TOP_K_CHUNKS = 5
CHUNK_PREVIEW_LENGTH = 100
MAX_TOKENS_ANSWER = 1024
MAX_TOKENS_SUMMARY = 64
CHUNK_NUMBERING_OFFSET = 1
```

**Lesson:** Named constants are self-documenting and easy to change.

---

### Note #4 — Missing HTTPException tests

**Problem:** No tests for missing API keys.

**Solution (`backend/tests/test_main.py`):**
```python
def test_get_embedding_service_raises_500_when_openai_key_missing() -> None:
    with patch("main._get_openai_api_key", return_value=None):
        with pytest.raises(HTTPException) as exc:
            get_embedding_service()
    assert exc.value.status_code == 500
    assert "OPENAI_API_KEY" in exc.value.detail
```

**Lesson:** Test error paths, not just the happy path.

---

### Note #5 — Missing edge case SSE tests

**Problem:** No tests for empty store and event sequence.

**Solution:**
```python
def test_stream_query_last_event_is_done() -> None:
    events = list(pipeline.stream_query("question?"))
    assert events[-1] == "event: done\ndata: {}\n\n"

def test_stream_query_empty_store_yields_valid_sse() -> None:
    pipeline = _make_streaming_pipeline(VectorStore(), tokens=["answer"])
    events = list(pipeline.stream_query("anything?"))
    assert events[0].startswith("event: chunks\n")
    assert events[-1] == "event: done\ndata: {}\n\n"
```

---

## 6. Architectural Decisions

### Next.js 14 App Router instead of Vite/React SPA

Next.js `output: "standalone"` generates a self-contained Node.js server ready for Dockerization.
App Router with `"use client"` allows granular decisions about which components are client-side.
Vite/React SPA requires a separate server (nginx) which complicates the Docker setup.

> `output: "standalone"` in `next.config.mjs` is the only change needed
> for a working Docker build of Next.js.

---

### FastAPI instead of Express/Django/Flask

FastAPI has native: Pydantic (validation), `Depends()` (DI), `StreamingResponse` (SSE),
async/await, automatic OpenAPI. Django is overengineering for a simple API without ORM.
Flask is synchronous and has no built-in DI. Express requires TypeScript and manual setup.

---

### NumPy cosine similarity instead of Pinecone/pgvector

For 29 chunks, NumPy linear search is instantaneous. Pinecone/pgvector
would add an external dependency (network, auth, cost) without benefit. NumPy allows
understanding how similarity search works — the educational goal of the project.

---

### Native fetch + SSE instead of Axios

Axios does not support SSE natively — it would require a separate library. Native `fetch` +
`ReadableStream` + `TextDecoder` gives full control over the SSE parser and zero
additional dependencies. Worth understanding this pattern deeply.

---

### Python dataclasses instead of dict/Pydantic

`@dataclass(frozen=True)` — immutable value objects with automatic `__repr__` and `__eq__`.
Pydantic is heavier and designed for validation at API boundaries (we use it there — in `main.py`).
Plain dicts are prone to typos in keys and have no typing.

---

### text-embedding-3-small model

Price-to-quality ratio: better than `ada-002`, cheaper than `text-embedding-3-large`.
1536 dimensions — sufficient for semantic search in a small knowledge base.
Multilingual — handles Polish questions about English documents.

---

## 7. Key Code Patterns to Remember

### Pattern 1 — SSE Streaming end-to-end

**Backend (generator):**
```python
# backend/rag/pipeline.py
def stream_query(self, question: str) -> Iterator[str]:
    metadata, prompt = self._retrieve(question)
    yield f"event: chunks\ndata: {json.dumps([...])}\n\n"
    with self._client.messages.stream(...) as stream:
        for text in stream.text_stream:
            yield f"event: token\ndata: {json.dumps(text)}\n\n"
    yield "event: done\ndata: {}\n\n"
```

**FastAPI endpoint:**
```python
# backend/main.py
@app.post("/query/stream")
def query_stream(request: QueryRequest, pipeline: RAGPipeline = Depends(get_pipeline)):
    return StreamingResponse(
        pipeline.stream_query(request.question),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

**Frontend parser:**
```typescript
// frontend/lib/api.ts
buffer += decoder.decode(value, { stream: true });
const events = buffer.split("\n\n");
buffer = events.pop() ?? "";
for (const event of events) {
  dispatchSSEEvent(type, data, callbacks);
}
```

> SSE format: `event: <type>\ndata: <json>\n\n` — two newlines at the end are the mandatory separator.

---

### Pattern 2 — Dependency Injection with FastAPI `Depends()`

```python
# backend/main.py — dependency chain
def get_embedding_service() -> EmbeddingService:
    api_key = _get_openai_api_key()
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
    return EmbeddingService(api_key=api_key)

def get_pipeline(
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    anthropic_client: anthropic.Anthropic = Depends(get_anthropic_client),
) -> RAGPipeline:
    return RAGPipeline(store=_store, embedding_service=embedding_service,
                       anthropic_client=anthropic_client)
```

**Override in tests:**
```python
app.dependency_overrides[get_pipeline] = lambda: mock_pipeline
try:
    response = client.post("/query/stream", json={"question": "test?"})
finally:
    app.dependency_overrides.clear()
```

---

### Pattern 3 — Cosine Similarity with NumPy (vectorization)

```python
# backend/rag/store.py
# Instead of a slow Python loop:
# scores = [np.dot(q, e) for e in embeddings]

# Matrix operation — all similarities at once:
matrix = np.stack([item.embedding for item in self._items])  # N × 1536
matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
query = query / np.linalg.norm(query)
scores = matrix @ query            # N results in one operation
top_indices = np.argsort(scores)[::-1][:top_k]
```

> NumPy vectorization vs Python loop — orders of magnitude faster for large N.

---

### Pattern 4 — Typewriter Effect Hook

```typescript
// frontend/hooks/useTypewriter.ts
export function useTypewriter(text: string, active: boolean): string {
  const [pos, setPos] = useState(0);
  const textRef = useRef(text);
  textRef.current = text;  // ← ref does not cause re-render, but is always current

  useEffect(() => {
    if (!active) return;
    const id = setInterval(() => {
      setPos(p => p + CHARS_PER_TICK >= textRef.current.length
        ? textRef.current.length : p + CHARS_PER_TICK);
    }, TICK_MS);
    return () => clearInterval(id);  // cleanup
  }, [active]);  // ← only one interval for the entire streaming

  if (!active) return text;
  return text.slice(0, pos);
}
```

> Use `useRef` instead of `useState` when you need the current value in a closure
> without creating a new `setInterval` on every change.

---

### Pattern 5 — Overlapping Chunking

```python
# backend/rag/chunker.py
step = self._chunk_size - self._overlap  # 100 - 50 = 50

for start in range(0, len(words), step):
    window = words[start : start + self._chunk_size]
    # Chunk 0: words 0-99
    # Chunk 1: words 50-149  ← 50 words shared with chunk 0
    # Chunk 2: words 100-199 ← 50 words shared with chunk 1
```

---

### Pattern 6 — lifespan Context Manager

```python
# backend/main.py
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # ← CODE ON STARTUP
    _ingest_documents(EmbeddingService(api_key=api_key))

    yield  # ← APPLICATION RUNS

    # ← CODE ON SHUTDOWN (cleanup)

app = FastAPI(lifespan=lifespan)
```

---

## 8. Job Interview Preparation — Q&A

### Q1: Describe step by step what happens when a user sends a question.

**Answer:**
1. User types a question in `ChatInput`, clicks Send
2. `useChat.sendMessage()` calls `queryStream()` from `lib/api.ts`
3. Frontend sends `POST /query/stream` with `{"question": "..."}` to FastAPI
4. FastAPI via `Depends()` creates `RAGPipeline` with injected services
5. `_retrieve()`: question is embedded via OpenAI, top-10 chunks
   searched by cosine similarity in NumPy VectorStore
6. Generator returns `event: chunks` — frontend displays fragments in sidebar
7. Pipeline opens a stream to Claude with a prompt containing the fragments
8. Each token from Claude → `event: token` → `useChat.onToken()` → `useTypewriter`
   displays characters smoothly every 18 ms
9. After completion → `event: done` → `isStreaming = false`

---

### Q2: How does your RAG pipeline prevent hallucinations?

**Answer:**
Two mechanisms. First, `SYSTEM_PROMPT` in `pipeline.py` contains an explicit instruction:
"Answer using ONLY the provided document excerpts. Do not invent information."
Second, only verified content from Markdown files reaches Claude's context.
The model physically cannot base its answer on knowledge outside the provided fragments.
If information is not in the documents, Claude is instructed to say so explicitly.

---

### Q3: Why did you choose cosine similarity instead of dot product?

**Answer:**
Dot product depends on vector length — a longer text would give a higher score just because of
its "magnitude", not semantic similarity. Cosine similarity measures the angle between
vectors regardless of their length. In practice, in `store.py` we normalize vectors
and use the `@` operation (dot product) — after normalization both measures are equivalent,
but semantically cosine is the more correct choice for retrieval.

---

### Q4: What is chunk overlap and why does it matter?

**Answer:**
Overlap is the number of words shared between adjacent chunks. In `chunker.py` we use
`CHUNK_SIZE_WORDS = 100` and `OVERLAP_WORDS = 50`, giving a step of 50 words. Without overlap
an important sentence could land exactly on the boundary — it would be cut in both chunks
and neither would contain the full context. Overlap = 50 guarantees that every sentence
appears complete in at least one chunk. Trade-off: increased number of chunks
and higher embedding cost.

---

### Q5: How does streaming work from backend to browser?

**Answer:**
Backend uses `StreamingResponse` from FastAPI and the `stream_query()` generator.
The generator produces SSE events in format `event: <type>\ndata: <json>\n\n`.
FastAPI passes these strings directly to TCP without buffering.
Frontend uses `fetch()` with `res.body.getReader()` — the browser's `ReadableStream` API.
Reads bytes in a loop, decodes via `TextDecoder` and parses by `\n\n` separator.
Each complete event is dispatched to a callback (`onChunks`, `onToken`, `onDone`).

---

### Q6: How would you replace the NumPy VectorStore with pgvector in production?

**Answer:**
`RAGPipeline` receives `VectorStore` via constructor (DIP) — it's enough to create
a `PgVectorStore` with the same `add()` and `search()` methods. In `add()`:
`INSERT INTO embeddings (text, source, embedding) VALUES (...)` (pgvector column).
In `search()`: `SELECT * FROM embeddings ORDER BY embedding <=> $1 LIMIT $2`
(cosine distance operator `<=>`). `RAGPipeline` requires no changes.
Additional benefit: persistence between restarts, HNSW index for O(log N) search.

---

### Q7: What SOLID principles did you apply and where?

**Answer:**
- **SRP** (`chunker.py`, `store.py`, `embeddings.py`): each class has one responsibility
- **OCP** (`VectorStore`): `PgVectorStore` can be added without modifying clients
- **DIP** (`pipeline.py`): `RAGPipeline` depends on types, not on concrete classes —
  all dependencies injected via constructor
- **DIP** (`main.py`): FastAPI `Depends()` as IoC container — framework
  automatically resolves the dependency tree

---

### Q8: How would you add conversation memory?

**Answer:**
Currently each question is independent. To add multi-turn: add a
`history: list[dict]` field to `QueryRequest`. Inject history in `build_prompt()`:
```python
history_text = "\n".join(f"{m['role']}: {m['content']}" for m in history[-3:])
prompt = f"...\n\nHistory:\n{history_text}\n\nQuestion: {question}"
```
Frontend `useChat` already stores `messages` — just pass the last
N messages. Persistence between sessions: PostgreSQL + session ID in cookies.

---

### Q9: What happens when the user asks in Polish and documents are in English?

**Answer:**
The `text-embedding-3-small` model is multilingual — it maps semantically similar
concepts in different languages to close vectors. "Ile dni urlopu" has high
similarity to "annual leave days". However, cross-lingual semantic gap can be
a problem for words differing in semantics (e.g. "komputer" vs "laptop").
Solution applied in the project: increasing `TOP_K_CHUNKS` from 5 to 10.
Production solution: query translation — translate the question to EN before embedding.
Claude's response is always in the question's language thanks to `SYSTEM_PROMPT`.

---

### Q10: How would you scale this to 10,000 documents?

**Answer:**
NumPy in-memory search is linear O(N). With 10K docs × ~20 chunks = 200K chunks,
RAM: ~200K × 1536 × 4 bytes ≈ 1.2 GB — acceptable, but search starts
to be noticeable. Solution: pgvector with HNSW index (O(log N), ~10 ms for 1M vectors).
An asynchronous ingestion pipeline (Celery/background tasks) is also needed because
embedding 200K chunks via OpenAI API takes minutes. Cache popular queries in Redis.

---

### Q11: Why FastAPI instead of Flask or Django?

**Answer:**
Flask is synchronous — each request blocks a thread, which with long LLM calls
(several seconds) limits throughput. Django is a full-stack framework with ORM,
templates, admin — overengineering for a simple REST API without a database.
FastAPI: async natively (`async def`), `Depends()` as IoC container, Pydantic
for request/response validation, `StreamingResponse` for SSE, automatic Swagger UI.
Everything needed without unnecessary layers.

---

### Q12: What is the role of the lifespan context manager in FastAPI?

**Answer:**
`lifespan` replaced the deprecated `@app.on_event("startup")`. It's an async context manager
that executes code before `yield` on application startup (here: auto-ingestion of documents
into VectorStore) and code after `yield` on shutdown (resource cleanup).
Key: it's wrapped in exception handling — if ingestion fails, the application starts
anyway (we log the exception, continue without indexed documents).
In Docker with `restart: unless-stopped` the container restarts and retries ingestion.

---

## 9. What I Would Do Differently in Production

### 1. Replace NumPy with pgvector

**Limitation:** In-memory — data lost on restart. Linear search O(N).
**Solution:** PostgreSQL + `pgvector` extension. HNSW index → O(log N).
Persistence, metadata filtering, scalability.

### 2. Semantic chunking

**Limitation:** Splitting every 100 words cuts sentences and paragraphs at random points.
**Solution:** `RecursiveCharacterTextSplitter` (LangChain) splitting on `\n\n`, `\n`,
sentences — preserves semantic units.

### 3. Hybrid Search (BM25 + vector)

**Limitation:** Pure vector search handles exact keywords,
proper nouns, and cross-lingual semantic gap poorly.
**Solution:** BM25 (keyword matching) + vector search with Reciprocal Rank Fusion (RRF).
pgvector + `pg_bm25` or Elasticsearch/OpenSearch.

### 4. Authentication

**Limitation:** Anyone with the URL incurs the company's API costs.
**Solution:** NextAuth.js (OAuth with Google Workspace) + middleware checking session.
JWT token in `Authorization` header verified by FastAPI.

### 5. Conversation persistence (multi-turn)

**Limitation:** Each question is independent — model does not remember session context.
**Solution:** Session ID in cookies, message history in PostgreSQL.
Last N messages appended to the prompt.

### 6. Query Translation for Cross-lingual Retrieval

**Limitation:** Polish questions may not match English fragments.
**Solution:** Before embedding — a quick call to Claude/GPT-4o-mini:
"Translate to English: {question}". Embed the translated question.

### 7. Audit Logging

**Limitation:** No logging of who asked, when, what, and which fragments were used.
**Solution:** Log every query to the database: `user_id`, `question`, `retrieved_chunks`,
`answer`, `timestamp`, `model`, `tokens_used`. Required for compliance (GDPR, audit).

### 8. Rate Limiting

**Limitation:** One user can generate unlimited API costs.
**Solution:** `slowapi` (FastAPI middleware) + Redis → e.g. 20 requests/minute.

### 9. Observability

**Limitation:** Logs only in container stdout — no metrics, tracing, alerts.
**Solution:** OpenTelemetry SDK → Datadog/Grafana. Metrics: query latency,
token cost per request, retrieval hit rate. Alerts when API cost exceeds threshold.

---

## 10. Glossary

| Term | Definition in project context |
|---|---|
| **RAG** | Retrieval-Augmented Generation — architecture where LLM answers questions based on documents retrieved in real time, not from model memory |
| **LLM** | Large Language Model — language model (here: Claude) generating text based on a prompt |
| **Embedding** | Text representation as a vector of numbers (1536 float32) — semantically similar texts have close vectors |
| **Vector** | List of floating-point numbers representing text in multidimensional space |
| **Cosine similarity** | Measure of similarity between two vectors (cosine of angle) — 1.0 = identical, 0.0 = orthogonal |
| **Chunking** | Division of long documents into smaller fragments for individual indexing |
| **Overlap** | Number of words shared between adjacent chunks — prevents information loss at boundaries |
| **Prompt injection** | Attack where user injects instructions for the LLM in the question content |
| **Context window** | Maximum number of tokens visible to the LLM — prompt + documents + question |
| **Hallucination** | LLM generates convincingly sounding but false information |
| **SSE** | Server-Sent Events — HTTP protocol for one-way streaming from server to client |
| **Streaming** | Sending responses in chunks in real time instead of waiting for the whole response |
| **SOLID** | 5 OOP principles: Single Responsibility, Open/Closed, Liskov, Interface Segregation, Dependency Inversion |
| **DRY** | Don't Repeat Yourself — every logic in one place, not duplicated |
| **Dependency Injection** | Pattern where an object receives its dependencies from outside instead of creating them itself |
| **lifespan event** | FastAPI context manager executing code on application startup and shutdown |
| **CORS** | Cross-Origin Resource Sharing — browser security mechanism blocking cross-origin requests |
| **Docker Compose** | Tool for running multiple containers as one service with a shared network |
| **Healthcheck** | `/health` endpoint returning application status — used by load balancers and monitoring |
| **Conventional commits** | Formatting standard: `feat:`, `fix:`, `chore:`, `docs:` — readable git history |
| **Top-k retrieval** | Fetching k most similar chunks (here: `TOP_K_CHUNKS = 10`) |
| **text-embedding-3-small** | OpenAI model generating embeddings — multilingual, 1536 dim, good price-to-quality ratio |
| **claude-sonnet-4** | Anthropic model used for generating answers in this project |
| **HNSW** | Hierarchical Navigable Small World — vector indexing algorithm, O(log N) search in pgvector |
| **IoC container** | Inversion of Control container — mechanism (here: `Depends()` in FastAPI) automatically resolving dependencies |
