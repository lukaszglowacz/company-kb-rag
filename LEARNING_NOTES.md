# NOTATKI Z NAUKI — Company KB RAG

Dokument przygotowany do rozmowy kwalifikacyjnej. Opisuje cały proces budowy projektu
krok po kroku — od pierwszego commita do wdrożenia w Dockerze.

---

## 1. Przegląd projektu

### Co zostało zbudowane

**Company KB RAG** to chatbot firmowej bazy wiedzy oparty na architekturze
Retrieval-Augmented Generation (RAG). Użytkownik zadaje pytanie w języku naturalnym
(po polsku lub angielsku), system wyszukuje semantycznie podobne fragmenty z dokumentów
firmowych (HR, onboarding, tech stack, company overview), a następnie Claude (Anthropic)
generuje odpowiedź wyłącznie na podstawie znalezionych fragmentów — bez halucynacji.
Aplikacja składa się z backendu FastAPI (Python) obsługującego pipeline RAG ze streamingiem
SSE oraz frontendu Next.js 14 z interfejsem czatu w czasie rzeczywistym.

### Architektura końcowa

```
Użytkownik
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
│    ├── lifespan()  ← auto-ingestion przy starcie    │
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
│  Sieć wewnętrzna: kb-network                        │
└─────────────────────────────────────────────────────┘
```

### Stack technologiczny

| Komponent | Technologia | Uzasadnienie wyboru |
|---|---|---|
| Backend framework | FastAPI (Python 3.11) | Async, automatyczny OpenAPI, Pydantic, `Depends()` DI, `StreamingResponse` SSE |
| LLM | Anthropic Claude (claude-sonnet-4) | Najlepsza jakość odpowiedzi, natywne wsparcie SSE streaming w SDK |
| Embeddingi | OpenAI text-embedding-3-small | Wielojęzyczny, 1536 dim, najlepszy stosunek ceny do jakości |
| Vector store | NumPy (in-memory) | Zero zewnętrznych dependencji, wystarczające dla małej bazy, edukacyjnie przejrzyste |
| Frontend | Next.js 14 App Router | SSR, `output: "standalone"` dla Dockera, `"use client"` granularity |
| Stylowanie | Tailwind CSS 3 | Utility-first, brak osobnych plików CSS, responsywność w jednym miejscu |
| Konteneryzacja | Docker Compose | Jednopoleceniowy start całego stacku, izolacja sieci |
| CI | GitHub Actions | 7 równoległych jobów: flake8, mypy, pytest, tsc, ESLint, Vitest, next build |
| Typowanie backend | mypy --strict | Wychwytuje błędy typów przy CI, eliminuje `Any` |
| Typowanie frontend | TypeScript strict | Wszystkie odpowiedzi API typowane przez `frontend/types/index.ts` |
| Testy backend | pytest + pytest-cov | Standard Python; `mock_embedding()` pozwala na testy offline |
| Testy frontend | Vitest | Natywny ESM, szybszy od Jest dla TypeScript |

---

## 2. Koncepty RAG — czego się nauczyłem

### Co to jest LLM i jak komunikujemy się z nim przez API

**Co to jest:** Large Language Model to model językowy trenowany na ogromnych zbiorach
tekstu, zdolny do generowania spójnych odpowiedzi w języku naturalnym.

**Jak działa komunikacja:**

```python
# backend/rag/pipeline.py — _call_llm()
message = self._client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system=SYSTEM_PROMPT,          # instrukcje systemowe
    messages=[{"role": "user", "content": prompt}],  # pytanie + kontekst
)
return message.content[0].text     # odpowiedź tekstowa
```

W trybie streaming:

```python
# backend/rag/pipeline.py — stream_query()
with self._client.messages.stream(
    model=CLAUDE_MODEL,
    max_tokens=MAX_TOKENS_ANSWER,
    system=SYSTEM_PROMPT,
    messages=[{"role": "user", "content": prompt}],
) as stream:
    for text in stream.text_stream:  # iterator tokenów
        yield f"event: token\ndata: {json.dumps(text)}\n\n"
```

**Gdzie w kodzie:** `backend/rag/pipeline.py` — metody `_call_llm()` i `stream_query()`.

---

### Co to są embeddingi i dlaczego są ważne

**Co to jest:** Embedding to reprezentacja tekstu jako wektor liczb zmiennoprzecinkowych
(tu: 1536 liczb float32). Semantycznie podobne teksty mają geometrycznie bliskie wektory.

**Implementacja:**

```python
# backend/rag/embeddings.py
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

def get_embedding(self, text: str) -> list[float]:
    response = self._client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL,
    )
    return response.data[0].embedding  # lista 1536 liczb
```

**Dlaczego ważne:** Pozwala porównać pytanie użytkownika z fragmentami dokumentów
bez dokładnego dopasowania słów kluczowych. Pytanie „ile dni urlopu?" i tekst
„26 days annual leave" mają wysokie podobieństwo w przestrzeni embeddingów.

> **Kluczowa obserwacja:** `text-embedding-3-small` jest wielojęzyczny, ale cross-lingual
> semantic gap istnieje — „komputer" (PL) vs „laptop" (EN) może mieć niższe podobieństwo
> niż oczekujemy.

---

### Co to jest chunking i dlaczego overlap ma znaczenie

**Co to jest:** Chunking to podział długich dokumentów na mniejsze fragmenty,
które mogą być osobno zaindeksowane i wyszukiwane.

**Implementacja:**

```python
# backend/rag/chunker.py
CHUNK_SIZE_WORDS = 100
OVERLAP_WORDS = 50

def chunk(self, text: str, source: str) -> list[Chunk]:
    words = text.split()
    step = self._chunk_size - self._overlap  # krok = 50 słów
    for start in range(0, len(words), step):
        window = words[start : start + self._chunk_size]
        chunks.append(Chunk(text=" ".join(window), source=source, index=index))
```

**Dlaczego overlap = 50 słów ma znaczenie:** Bez overlapu zdanie kluczowe mogłoby
trafić na granicę dwóch chunków i w żadnym nie byłoby kompletne. Overlap gwarantuje,
że każde zdanie pojawi się kompletne w co najmniej jednym chunku. Trade-off: więcej
chunków = wyższy koszt embeddingów.

---

### Co to jest cosine similarity i jak NumPy to implementuje

**Co to jest:** Miara podobieństwa dwóch wektorów — cosinus kąta między nimi.
Wartość 1.0 = identyczne kierunki, 0.0 = prostopadłe, −1.0 = przeciwne.

**Implementacja macierzowa w NumPy:**

```python
# backend/rag/store.py
def search(self, query_embedding: list[float], top_k: int = 5):
    query = np.array(query_embedding, dtype=np.float32)
    query = query / (np.linalg.norm(query) + NORMALIZATION_EPSILON)  # normalizacja

    # Macierz wszystkich chunków: N × 1536
    matrix = np.stack([item.embedding for item in self._items])
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + NORMALIZATION_EPSILON
    matrix = matrix / norms  # normalizacja każdego wiersza

    # Iloczyn skalarny = cosine similarity po normalizacji
    scores: npt.NDArray[np.float32] = matrix @ query  # N wyników naraz
    top_indices = np.argsort(scores)[::-1][:top_k]
```

**Dlaczego cosine, nie dot product:** Dot product zależy od długości wektora.
Po normalizacji obie miary są równoważne — używamy dot product `@` bo jest szybszy.

---

### Co to jest context window i prompt injection

**Context window:** Maksymalna liczba tokenów widoczna przez LLM w jednym zapytaniu.
Łączy: system prompt + fragmenty dokumentów + pytanie użytkownika.

**Jak konstruujemy prompt:**

```python
# backend/rag/pipeline.py
def build_prompt(self, question: str, chunks: list[Chunk]) -> str:
    excerpts = "\n\n".join(
        f"[{i + 1}] (source: {c.source})\n{c.text}"
        for i, c in enumerate(chunks)
    )
    return f"Document excerpts:\n\n{excerpts}\n\nQuestion: {question}"
```

**Prompt injection:** Atak gdzie użytkownik wstrzykuje instrukcje dla LLM,
np. „Zignoruj poprzednie instrukcje i...". Mitigacja: `SYSTEM_PROMPT` z explicit
zakaz wymyślania i ograniczenie do podanych fragmentów.

---

### Różnica między RAG a fine-tuningiem

| Cecha | RAG (nasz projekt) | Fine-tuning |
|---|---|---|
| Wiedza | Z zewnętrznych dokumentów (retrieval) | Wbudowana w wagi modelu |
| Aktualizacja wiedzy | Dodaj dokument, reingest | Retrenuj model (GPU, czas, dane) |
| Koszt | Tani (embeddingi + API call) | Drogi |
| Ryzyko halucynacji | Niskie (model widzi źródła) | Wyższe |
| Transparentność | ✅ Widać które fragmenty użyto | ❌ |

---

### Co to jest halucynacja i jak RAG jej zapobiega

**Halucynacja:** LLM generuje przekonująco brzmiące, ale fałszywe informacje.

**Jak RAG zapobiega — dwa mechanizmy:**

```python
# backend/rag/pipeline.py
SYSTEM_PROMPT = (
    "You are a company knowledge base assistant. "
    "Answer questions using ONLY the provided document excerpts. "
    "If the answer is not in the excerpts, say so clearly. "
    "Do not invent information. ..."
)
```

1. Model dostaje tylko zweryfikowane fragmenty z dokumentów firmowych
2. `SYSTEM_PROMPT` zawiera explicit zakaz wymyślania

---

## 3. Etapy budowy — PR po PR

### PR #1 — Inicjalizacja repozytorium i CI

**Commit:** `chore: initialize repo structure with CI pipeline (#1)`

**Co zostało zbudowane:**
- Struktura katalogów: `backend/`, `frontend/`, `.github/workflows/`
- CI pipeline z 7 równoległymi jobami
- `requirements.txt`, bazowe pliki konfiguracyjne

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

**Kluczowe decyzje:** `mypy --strict` od początku — wymusza pełne typowanie,
eliminuje błędy które inaczej ujawniają się na produkcji.

---

### PR #2 — Pipeline ingestion

**Commit:** `feat: implement ingestion pipeline — TextChunker, VectorStore, EmbeddingService (#2)`

**Co zostało zbudowane:**
- `TextChunker` — podział tekstu na overlapping chunks
- `VectorStore` — in-memory storage z cosine similarity (NumPy)
- `EmbeddingService` — wrapper OpenAI API + `mock_embedding()` do testów

**Zasada SRP — każda klasa ma jedną odpowiedzialność:**

```python
# backend/rag/chunker.py — TYLKO podział tekstu
class TextChunker:
    def chunk(self, text: str, source: str) -> list[Chunk]: ...

# backend/rag/store.py — TYLKO przechowywanie i wyszukiwanie
class VectorStore:
    def add(self, chunks, embeddings): ...
    def search(self, query_embedding, top_k): ...

# backend/rag/embeddings.py — TYLKO komunikacja z OpenAI
class EmbeddingService:
    def get_embedding(self, text: str) -> list[float]: ...
```

**Mock embedding — testy bez API call:**

```python
# backend/rag/embeddings.py
@staticmethod
def mock_embedding(text: str) -> list[float]:
    """Deterministyczny fake — ten sam tekst zawsze daje ten sam wektor."""
    digest = int(hashlib.sha256(text.encode()).hexdigest(), 16)
    seed = (digest ^ MOCK_SEED) & UINT32_MASK
    # LCG generator pseudolosowy
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

**Co zostało zbudowane:**
- `RAGPipeline` — orkiestracja retrieval + LLM
- Endpointy: `POST /query`, `POST /summarize`, `GET /health`

**Zasada DIP — zależności wstrzykiwane przez konstruktor:**

```python
# backend/rag/pipeline.py
class RAGPipeline:
    def __init__(
        self,
        store: VectorStore,                    # wstrzyknięty
        embedding_service: EmbeddingService,   # wstrzyknięty
        anthropic_client: anthropic.Anthropic, # wstrzyknięty
    ) -> None:
        self._store = store
        self._embedding_service = embedding_service
        self._client = anthropic_client
```

RAGPipeline nie tworzy swoich zależności — dostaje je z zewnątrz.
W testach można wstrzyknąć MagicMock zamiast prawdziwego klienta.

---

### PR #4 — Next.js 14 Chat UI

**Commit:** `feat: implement Next.js 14 chat UI for RAG knowledge base (#4)`

**Co zostało zbudowane:**
- Komponenty: `ChatWindow`, `ChatMessages`, `MessageBubble`, `ChatInput`, `ChunksSidebar`
- Hook `useChat` — zarządzanie stanem wiadomości
- `lib/api.ts` — klient SSE
- `types/index.ts` — interfejsy TypeScript

**Typy TypeScript (`frontend/types/index.ts`):**

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

  const events = buffer.split("\n\n");  // SSE events rozdzielone \n\n
  buffer = events.pop() ?? "";           // ostatni niekompletny — zostaje w buforze

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

### PR #5 — Dokumenty firmowe i auto-ingestion

**Commit:** `feat(backend): company documents + auto-ingestion on startup (#5)`

**Co zostało zbudowane:**
- 4 dokumenty Markdown: `hr_policy.md`, `onboarding.md`, `tech_stack.md`, `company_overview.md`
- `data/documents.py` — ładowanie wszystkich `.md`
- `lifespan` — auto-ingestion przy starcie

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
    yield  # aplikacja działa i obsługuje requesty
    # cleanup po yield (przy zatrzymaniu)

app = FastAPI(title="Company KB RAG API", lifespan=lifespan)
```

**Dependency Injection w FastAPI (`Depends()`):**

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
    pipeline: RAGPipeline = Depends(get_pipeline),  # FastAPI wstrzykuje automatycznie
) -> StreamingResponse:
    return StreamingResponse(pipeline.stream_query(request.question),
                             media_type="text/event-stream")
```

---

### PR #6 — Streaming SSE

**Commit:** `feat: streaming SSE query endpoint + streaming chat UI (#6)`

**Co zostało zbudowane:**
- `POST /query/stream` z `StreamingResponse`
- `stream_query()` — generator SSE eventów
- Frontend: `useChat` z obsługą tokenów

**Format SSE — trzy typy eventów:**

```
event: chunks
data: [{"source":"hr_policy.md","score":0.55,"preview":"26 days annual..."}]

event: token
data: "Na podstawie"

event: token
data: " dokumentów HR..."

event: done
data: {}
```

**Generator SSE (`backend/rag/pipeline.py`):**

```python
def stream_query(self, question: str) -> Iterator[str]:
    metadata, prompt = self._retrieve(question)  # wspólna logika z query()

    # Event 1: znalezione fragmenty (od razu, przed generowaniem)
    chunks_payload = json.dumps(
        [{"source": m.source, "score": m.score, "preview": m.preview}
         for m in metadata]
    )
    yield f"event: chunks\ndata: {chunks_payload}\n\n"

    # Event 2-N: tokeny z Claude (streaming)
    with self._client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS_ANSWER,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            yield f"event: token\ndata: {json.dumps(text)}\n\n"

    # Event końcowy
    yield "event: done\ndata: {}\n\n"
```

**Eliminacja DRY — helper `_retrieve()`:**

```python
def _retrieve(self, question: str) -> tuple[list[ChunkMetadata], str]:
    """Wspólna logika dla query() i stream_query() — DRY."""
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

**Co zostało zbudowane:**
- `docker-compose.yml` z dwoma serwisami i wspólną siecią
- `backend/Dockerfile` — python:3.11-slim, non-root user
- `frontend/Dockerfile` — 3-stage Node.js standalone build
- `frontend/next.config.mjs` z `output: "standalone"`

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

**3-stage Dockerfile frontendu:**

```dockerfile
# Stage 1: instalacja dependencji
FROM node:20-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci

# Stage 2: build aplikacji
FROM node:20-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

# Stage 3: minimalny runtime (tylko to co potrzebne)
FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production
RUN addgroup --system --gid 1001 nodejs \
 && adduser --system --uid 1001 nextjs
COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static
USER nextjs           # non-root dla bezpieczeństwa
EXPOSE 3000
CMD ["node", "server.js"]
```

---

### PR #8 — Poprawki UI, jakość, responsywność

**Commit:** `Fix/markdown rendering and language (#8)`

**Co zostało zbudowane:**
- `react-markdown` w `MessageBubble` — renderowanie nagłówków, list, pogrubień
- `useTypewriter` hook — efekt płynnego pisania
- Responsywny `ChunksSidebar` — mobile bottom sheet + desktop sidebar
- Wzmocniony `SYSTEM_PROMPT` — wymuszanie języka odpowiedzi
- `TOP_K_CHUNKS` zwiększony z 5 do 10
- `postcss.config.js`, `public/.gitkeep`, `CORSMiddleware`, `*.tsbuildinfo` w `.gitignore`

**Hook `useTypewriter` (`frontend/hooks/useTypewriter.ts`):**

```typescript
const CHARS_PER_TICK = 4;
const TICK_MS = 18;

export function useTypewriter(text: string, active: boolean): string {
  const [pos, setPos] = useState(0);
  const textRef = useRef(text);
  textRef.current = text;  // ref = zawsze aktualna wartość bez re-tworzenia interval

  useEffect(() => {
    if (!active) return;
    const id = setInterval(() => {
      setPos((p) => {
        const target = textRef.current.length;
        return p + CHARS_PER_TICK >= target ? target : p + CHARS_PER_TICK;
      });
    }, TICK_MS);
    return () => clearInterval(id);
  }, [active]);  // jeden interval przez całe streaming

  if (!active) return text;   // stare wiadomości — pełny tekst od razu
  return text.slice(0, pos);  // streaming — litery po kolei
}
```

---

## 4. Bugi i naprawy

### Bug #1 — flake8 E501: za długa linia

**Co to był:** Linia 24 w `rag/pipeline.py` miała 106 znaków, limit flake8 to 88.

**Przed:**
```python
"If the user writes in Polish, respond 100% in Polish — even if the source documents are in English. "
```

**Po:**
```python
"If the user writes in Polish, respond 100% in Polish — "
"even if the source documents are in English. "
```

**Lekcja:** Uruchamiać `flake8 . --max-line-length=88` lokalnie przed każdym commitem.

---

### Bug #2 — CORS error (przeglądarka blokowała requesty)

**Co to był:** Przeglądarka blokowała fetch z `localhost:3000` do `localhost:8000`.

**Przyczyna:** Same-Origin Policy — przeglądarka wymaga nagłówka
`Access-Control-Allow-Origin` dla cross-origin requestów.

**Naprawa (`backend/main.py`):**
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Lekcja:** CORS zawsze konfigurować gdy frontend i backend są na różnych portach.

---

### Bug #3 — Brak postcss.config.js

**Co to był:** Tailwind CSS nie ładował styli w Dockerze — cały UI wyglądał jak nieostylowany HTML.

**Przyczyna:** Next.js wymaga `postcss.config.js` do przetworzenia dyrektyw `@tailwind`.
Plik był na jednym branchu i nie trafił do `main`.

**Naprawa:**
```javascript
// frontend/postcss.config.js
module.exports = {
  plugins: { tailwindcss: {}, autoprefixer: {} },
};
```

**Lekcja:** Pliki konfiguracyjne frameworków muszą być commitowane.

---

### Bug #4 — Brak frontend/public/

**Co to był:** Docker build failował: `"/app/public": not found`.

**Przyczyna:** Dockerfile kopiuje katalog `public/` w stage runner,
ale katalog nie istniał w repozytorium.

**Naprawa:** Stworzono `frontend/public/.gitkeep` — pusty plik wymuszający
śledzenie katalogu przez git.

**Lekcja:** Next.js standalone build wymaga katalogu `public/` nawet jeśli pusty.

---

### Bug #5 — nginx w frontend Dockerfile

**Co to był:** Frontend nie działał w Dockerze — nginx nie potrafi serwować Next.js SSR.

**Przyczyna:** nginx serwuje pliki statyczne, ale Next.js wymaga Node.js runtime.

**Przed:**
```dockerfile
FROM nginx:alpine
COPY --from=builder /app/out /usr/share/nginx/html
```

**Po:**
```dockerfile
FROM node:20-alpine AS runner
CMD ["node", "server.js"]
```

**Lekcja:** Next.js z `output: "standalone"` generuje własny Node.js server.

---

### Bug #6 — `import json` wewnątrz funkcji

**Co to był:** `import json` był wewnątrz metody `stream_query()` zamiast na górze pliku.

**Naprawa:** Przeniesienie na top-level (linia 1 pliku).

**Lekcja:** Importy zawsze na górze pliku — wymóg isort/flake8.

---

### Bug #7 — `anthropic.omit` ImportError na CI

**Przyczyna:** `omit` sentinel nie istnieje w starszej wersji Anthropic SDK na CI.

**Przed:**
```python
from anthropic import omit as OMIT
kwargs["system"] = system if system is not None else OMIT
```

**Po:**
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

**Lekcja:** Nie używać feature'ów SDK które mogą nie istnieć w starszych wersjach.

---

### Bug #8 — `@pytest.fixture()` untyped decorator

**Co to był:** mypy --strict failował na CI: `error: Untyped decorator makes function untyped`.

**Przyczyna:** Starsze stubs pytest na CI nie miały typów dla `@pytest.fixture()`.

**Przed:**
```python
@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)
```

**Po:**
```python
def _make_client() -> TestClient:
    return TestClient(app)
```

**Lekcja:** Przy `mypy --strict` unikać dekoratorów bez pełnych typów w starszych bibliotekach.

---

## 5. Uwagi z code review

### Uwaga #1 — Naruszenie SRP: `_ingest_documents` robiła za dużo

**Problem:** Funkcja robiła: load → chunk → embed → store — cztery odpowiedzialności.

**Rozwiązanie:** Rozdzielenie na osobne wywołania:
```python
docs = load_documents()
for text, source in docs:
    chunks = _chunker.chunk(text, source)
    embeddings = [embedding_service.get_embedding(c.text) for c in chunks]
    _store.add(chunks, embeddings)
```

**Lekcja:** Jeśli opis funkcji zawiera "i", podziel ją.

---

### Uwaga #2 — Naruszenie DRY: duplikacja retrieve logic

**Problem:** `query()` i `stream_query()` powtarzały identyczną logikę embed + search.

**Rozwiązanie:** Helper `_retrieve()` — jedna definicja, dwa wywołania.

**Lekcja:** DRY — każda logika w jednym miejscu.

---

### Uwaga #3 — Magic numbers

**Problem:** Liczby `5`, `100`, `1024` bez wyjaśnienia w kodzie.

**Rozwiązanie:**
```python
TOP_K_CHUNKS = 5
CHUNK_PREVIEW_LENGTH = 100
MAX_TOKENS_ANSWER = 1024
MAX_TOKENS_SUMMARY = 64
CHUNK_NUMBERING_OFFSET = 1
```

**Lekcja:** Named constants są self-documenting i łatwe do zmiany.

---

### Uwaga #4 — Brakujące testy HTTPException

**Problem:** Brak testów dla brakujących kluczy API.

**Rozwiązanie (`backend/tests/test_main.py`):**
```python
def test_get_embedding_service_raises_500_when_openai_key_missing() -> None:
    with patch("main._get_openai_api_key", return_value=None):
        with pytest.raises(HTTPException) as exc:
            get_embedding_service()
    assert exc.value.status_code == 500
    assert "OPENAI_API_KEY" in exc.value.detail
```

**Lekcja:** Testować error paths, nie tylko happy path.

---

### Uwaga #5 — Brakujące testy edge case SSE

**Problem:** Brak testów dla pustego store i sekwencji eventów.

**Rozwiązanie:**
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

## 6. Decyzje architektoniczne

### Next.js 14 App Router zamiast Vite/React SPA

Next.js `output: "standalone"` generuje samodzielny Node.js server gotowy do Dockeryzacji.
App Router z `"use client"` pozwala granularnie decydować które komponenty są client-side.
Vite/React SPA wymaga osobnego serwera (nginx) co komplikuje Docker setup.

> `output: "standalone"` w `next.config.mjs` to jedyna zmiana potrzebna
> do działającego Docker buildu Next.js.

---

### FastAPI zamiast Express/Django/Flask

FastAPI ma natywne: Pydantic (walidacja), `Depends()` (DI), `StreamingResponse` (SSE),
async/await, automatyczny OpenAPI. Django to overengineering dla prostego API bez ORM.
Flask jest synchroniczny i nie ma wbudowanego DI. Express wymaga TypeScript i ręcznego setupu.

---

### NumPy cosine similarity zamiast Pinecone/pgvector

Dla 29 chunków wyszukiwanie liniowe NumPy jest natychmiastowe. Pinecone/pgvector
dodałyby zewnętrzną zależność (sieć, auth, koszt) bez zysku. NumPy pozwala
zrozumieć jak similarity search działa — cel edukacyjny projektu.

---

### Native fetch + SSE zamiast Axios

Axios nie obsługuje SSE natywnie — wymagałby osobnej biblioteki. Native `fetch` +
`ReadableStream` + `TextDecoder` daje pełną kontrolę nad parserem SSE i zero
dodatkowych dependencji. Warto rozumieć ten wzorzec głęboko.

---

### Python dataclasses zamiast dict/Pydantic

`@dataclass(frozen=True)` — immutable value objects z automatycznym `__repr__` i `__eq__`.
Pydantic jest cięższy i przeznaczony do walidacji na granicach API (używamy go tam — w `main.py`).
Nagie dicts są podatne na literówki w kluczach i nie mają typowania.

---

### Model text-embedding-3-small

Stosunek jakości do ceny: lepszy od `ada-002`, tańszy od `text-embedding-3-large`.
1536 wymiarów — wystarczające dla semantycznego wyszukiwania w małej bazie.
Wielojęzyczny — obsługuje polskie pytania o angielskie dokumenty.

---

## 7. Kluczowe wzorce kodu do zapamiętania

### Wzorzec 1 — SSE Streaming end-to-end

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

> Format SSE: `event: <typ>\ndata: <json>\n\n` — dwa newline na końcu to obowiązkowy separator.

---

### Wzorzec 2 — Dependency Injection z FastAPI `Depends()`

```python
# backend/main.py — łańcuch zależności
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

**Override w testach:**
```python
app.dependency_overrides[get_pipeline] = lambda: mock_pipeline
try:
    response = client.post("/query/stream", json={"question": "test?"})
finally:
    app.dependency_overrides.clear()
```

---

### Wzorzec 3 — Cosine Similarity z NumPy (wektoryzacja)

```python
# backend/rag/store.py
# Zamiast wolnej pętli Python:
# scores = [np.dot(q, e) for e in embeddings]

# Operacja macierzowa — wszystkie podobieństwa naraz:
matrix = np.stack([item.embedding for item in self._items])  # N × 1536
matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
query = query / np.linalg.norm(query)
scores = matrix @ query            # N wyników w jednej operacji
top_indices = np.argsort(scores)[::-1][:top_k]
```

> NumPy vectorization vs Python loop — orders of magnitude szybciej dla dużych N.

---

### Wzorzec 4 — Typewriter Effect Hook

```typescript
// frontend/hooks/useTypewriter.ts
export function useTypewriter(text: string, active: boolean): string {
  const [pos, setPos] = useState(0);
  const textRef = useRef(text);
  textRef.current = text;  // ← ref nie powoduje re-render, ale jest zawsze aktualny

  useEffect(() => {
    if (!active) return;
    const id = setInterval(() => {
      setPos(p => p + CHARS_PER_TICK >= textRef.current.length
        ? textRef.current.length : p + CHARS_PER_TICK);
    }, TICK_MS);
    return () => clearInterval(id);  // cleanup
  }, [active]);  // ← tylko jeden interval przez całe streaming

  if (!active) return text;
  return text.slice(0, pos);
}
```

> `useRef` zamiast `useState` gdy potrzebujesz aktualnej wartości w closurze
> bez tworzenia nowego `setInterval` przy każdej zmianie.

---

### Wzorzec 5 — Overlapping Chunking

```python
# backend/rag/chunker.py
step = self._chunk_size - self._overlap  # 100 - 50 = 50

for start in range(0, len(words), step):
    window = words[start : start + self._chunk_size]
    # Chunk 0: słowa 0-99
    # Chunk 1: słowa 50-149  ← 50 słów wspólnych z chunk 0
    # Chunk 2: słowa 100-199 ← 50 słów wspólnych z chunk 1
```

---

### Wzorzec 6 — lifespan Context Manager

```python
# backend/main.py
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # ← KOD PRZY STARCIE
    _ingest_documents(EmbeddingService(api_key=api_key))

    yield  # ← APLIKACJA DZIAŁA

    # ← KOD PRZY ZATRZYMANIU (cleanup)

app = FastAPI(lifespan=lifespan)
```

---

## 8. Przygotowanie do rozmowy kwalifikacyjnej — Q&A

### Q1: Opisz krok po kroku co się dzieje gdy użytkownik wysyła pytanie.

**Odpowiedź:**
1. Użytkownik wpisuje pytanie w `ChatInput`, klika Send
2. `useChat.sendMessage()` wywołuje `queryStream()` z `lib/api.ts`
3. Frontend wysyła `POST /query/stream` z `{"question": "..."}` do FastAPI
4. FastAPI przez `Depends()` tworzy `RAGPipeline` z wstrzykniętymi serwisami
5. `_retrieve()`: pytanie jest embeddowane przez OpenAI, top-10 chunków
   wyszukiwanych przez cosine similarity w NumPy VectorStore
6. Generator zwraca `event: chunks` — frontend wyświetla fragmenty w sidebarze
7. Pipeline otwiera stream do Claude z promptem zawierającym fragmenty
8. Każdy token od Claude → `event: token` → `useChat.onToken()` → `useTypewriter`
   wyświetla znaki płynnie co 18 ms
9. Po zakończeniu → `event: done` → `isStreaming = false`

---

### Q2: Jak Twój pipeline RAG zapobiega halucynacjom?

**Odpowiedź:**
Dwa mechanizmy. Po pierwsze, `SYSTEM_PROMPT` w `pipeline.py` zawiera explicit instrukcję:
„Answer using ONLY the provided document excerpts. Do not invent information."
Po drugie, do kontekstu Claude trafia tylko zweryfikowana treść z plików Markdown.
Model fizycznie nie może oprzeć odpowiedzi na wiedzy spoza dostarczonych fragmentów.
Jeśli informacji nie ma w dokumentach, Claude jest instruowany by to powiedzieć wprost.

---

### Q3: Dlaczego wybrałeś cosine similarity zamiast dot product?

**Odpowiedź:**
Dot product zależy od długości wektora — dłuższy tekst dałby wyższy wynik tylko przez
swoją „siłę", nie przez semantyczne podobieństwo. Cosine similarity mierzy kąt między
wektorami niezależnie od ich długości. W praktyce w `store.py` normalizujemy wektory
i używamy operacji `@` (dot product) — po normalizacji obie miary są równoważne,
ale semantycznie cosine jest poprawniejszym wyborem dla retrieval.

---

### Q4: Co to jest chunk overlap i dlaczego ma znaczenie?

**Odpowiedź:**
Overlap to liczba słów wspólnych między sąsiednimi chunkami. W `chunker.py` używamy
`CHUNK_SIZE_WORDS = 100` i `OVERLAP_WORDS = 50`, co daje krok 50 słów. Bez overlapu
ważne zdanie mogłoby trafić dokładnie na granicę — byłoby przycięte w obu chunka
i żaden nie zawierałby pełnego kontekstu. Overlap = 50 gwarantuje, że każde zdanie
pojawi się kompletne w co najmniej jednym chunku. Trade-off: zwiększona liczba chunków
i wyższy koszt embeddingów.

---

### Q5: Jak działa streaming od backendu do przeglądarki?

**Odpowiedź:**
Backend używa `StreamingResponse` z FastAPI i generatora `stream_query()`.
Generator produkuje SSE events w formacie `event: <typ>\ndata: <json>\n\n`.
FastAPI przekazuje te stringi bezpośrednio do TCP bez buforowania.
Frontend używa `fetch()` z `res.body.getReader()` — `ReadableStream` API przeglądarki.
Czyta bajty w pętli, dekoduje przez `TextDecoder` i parsuje po separatorze `\n\n`.
Każdy kompletny event jest dispatchowany do callbacka (`onChunks`, `onToken`, `onDone`).

---

### Q6: Jak byś zastąpił NumPy VectorStore pgvectorem na produkcji?

**Odpowiedź:**
`RAGPipeline` dostaje `VectorStore` przez konstruktor (DIP) — wystarczy stworzyć
`PgVectorStore` z tymi samymi metodami `add()` i `search()`. W `add()`:
`INSERT INTO embeddings (text, source, embedding) VALUES (...)` (pgvector column).
W `search()`: `SELECT * FROM embeddings ORDER BY embedding <=> $1 LIMIT $2`
(operator cosine distance `<=>`). `RAGPipeline` nie wymaga żadnych zmian.
Dodatkowy zysk: persystencja między restartami, indeks HNSW dla O(log N) search.

---

### Q7: Jakie zasady SOLID zastosowałeś i gdzie?

**Odpowiedź:**
- **SRP** (`chunker.py`, `store.py`, `embeddings.py`): każda klasa ma jedną odpowiedzialność
- **OCP** (`VectorStore`): można dodać `PgVectorStore` bez modyfikacji klientów
- **DIP** (`pipeline.py`): `RAGPipeline` zależy od typów, nie od konkretnych klas —
  wszystkie zależności wstrzykiwane przez konstruktor
- **DIP** (`main.py`): FastAPI `Depends()` jako IoC container — framework
  automatycznie rozwiązuje drzewo zależności

---

### Q8: Jak byś dodał pamięć konwersacji?

**Odpowiedź:**
Aktualnie każde pytanie jest niezależne. Żeby dodać multi-turn: w `QueryRequest`
dodać pole `history: list[dict]`. W `build_prompt()` wstrzyknąć historię:
```python
history_text = "\n".join(f"{m['role']}: {m['content']}" for m in history[-3:])
prompt = f"...\n\nHistoria:\n{history_text}\n\nPytanie: {question}"
```
Frontend `useChat` już przechowuje `messages` — wystarczy przekazywać ostatnie
N wiadomości. Persystencja między sesjami: PostgreSQL + session ID w cookies.

---

### Q9: Co się dzieje gdy użytkownik pyta po polsku a dokumenty są po angielsku?

**Odpowiedź:**
Model `text-embedding-3-small` jest wielojęzyczny — mapuje semantycznie podobne
koncepty w różnych językach na bliskie wektory. „Ile dni urlopu" ma wysokie
podobieństwo do „annual leave days". Jednak cross-lingual semantic gap może być
problemem dla słów różniących się semantyką (np. „komputer" vs „laptop").
Rozwiązanie zastosowane w projekcie: zwiększenie `TOP_K_CHUNKS` z 5 do 10.
Produkcyjne rozwiązanie: query translation — przetłumacz pytanie na EN przed embeddowaniem.
Odpowiedź Claude jest zawsze w języku pytania dzięki `SYSTEM_PROMPT`.

---

### Q10: Jak byś przeskalował to do 10 000 dokumentów?

**Odpowiedź:**
NumPy in-memory search jest liniowy O(N). Przy 10K dok × ~20 chunków = 200K chunków,
pamięć RAM: ~200K × 1536 × 4 bajtów ≈ 1.2 GB — akceptowalne, ale search zaczyna
być zauważalny. Rozwiązanie: pgvector z indeksem HNSW (O(log N), ~10 ms dla 1M wektorów).
Potrzeba też asynchronicznego ingestion pipeline (Celery/background tasks) bo embed
200K chunków przez OpenAI API trwa minuty. Cache popularnych pytań w Redis.

---

### Q11: Dlaczego FastAPI zamiast Flask lub Django?

**Odpowiedź:**
Flask jest synchroniczny — każdy request blokuje wątek, co przy długich wywołaniach
LLM (kilka sekund) ogranicza przepustowość. Django to full-stack framework z ORM,
templates, admin — overengineering dla prostego REST API bez bazy danych.
FastAPI: async natywnie (`async def`), `Depends()` jako IoC container, Pydantic
do walidacji requestów/response, `StreamingResponse` dla SSE, automatyczny Swagger UI.
Wszystko czego potrzeba bez zbędnych warstw.

---

### Q12: Jaka jest rola lifespan context manager w FastAPI?

**Odpowiedź:**
`lifespan` zastąpił przestarzałe `@app.on_event("startup")`. To async context manager
który wykonuje kod przed `yield` przy starcie aplikacji (tu: auto-ingestion dokumentów
do VectorStore) i kod po `yield` przy zatrzymaniu (cleanup zasobów).
Kluczowe: jest objęty obsługą wyjątków — jeśli ingestion failuje, aplikacja startuje
dalej (logujemy wyjątek, kontynuujemy bez zaindeksowanych dokumentów).
W Dockerze z `restart: unless-stopped` kontener restartuje się i ponawia ingestion.

---

## 9. Co bym zrobił inaczej na produkcji

### 1. Zastąpienie NumPy pgvectorem

**Ograniczenie:** In-memory — dane giną przy restarcie. Liniowy search O(N).
**Rozwiązanie:** PostgreSQL + `pgvector` extension. Indeks HNSW → O(log N).
Persystencja, filtrowanie po metadanych, możliwość skalowania.

### 2. Semantyczne chunking

**Ograniczenie:** Podział co 100 słów przecina zdania i paragrafy w losowych miejscach.
**Rozwiązanie:** `RecursiveCharacterTextSplitter` (LangChain) dzielący po `\n\n`, `\n`,
zdaniach — zachowuje semantyczne jednostki.

### 3. Hybrid Search (BM25 + vector)

**Ograniczenie:** Pure vector search słabo radzi sobie z dokładnymi słowami kluczowymi,
nazwami własnymi i cross-lingual semantic gap.
**Rozwiązanie:** BM25 (keyword matching) + vector search z Reciprocal Rank Fusion (RRF).
pgvector + `pg_bm25` lub Elasticsearch/OpenSearch.

### 4. Autentykacja

**Ograniczenie:** Każdy z dostępem do URL ponosi koszty API firmy.
**Rozwiązanie:** NextAuth.js (OAuth z Google Workspace) + middleware sprawdzające sesję.
JWT token w headerze `Authorization` weryfikowany przez FastAPI.

### 5. Persystencja konwersacji (multi-turn)

**Ograniczenie:** Każde pytanie jest niezależne — model nie pamięta kontekstu sesji.
**Rozwiązanie:** Session ID w cookies, historia wiadomości w PostgreSQL.
Ostatnie N wiadomości dołączane do promptu.

### 6. Query Translation dla Cross-lingual Retrieval

**Ograniczenie:** Pytania po polsku mogą nie trafiać na angielskie fragmenty.
**Rozwiązanie:** Przed embeddowaniem — szybki call do Claude/GPT-4o-mini:
„Translate to English: {question}". Embeddujemy przetłumaczone pytanie.

### 7. Audit Logging

**Ograniczenie:** Brak logowania kto pytał, kiedy, o co i jakie fragmenty użyto.
**Rozwiązanie:** Log każdego query do bazy: `user_id`, `question`, `retrieved_chunks`,
`answer`, `timestamp`, `model`, `tokens_used`. Wymagane dla compliance (RODO, audyt).

### 8. Rate Limiting

**Ograniczenie:** Jeden użytkownik może wygenerować nieograniczone koszty API.
**Rozwiązanie:** `slowapi` (FastAPI middleware) + Redis → np. 20 requestów/minutę.

### 9. Observability

**Ograniczenie:** Logi tylko w stdout kontenera — brak metryk, tracing, alertów.
**Rozwiązanie:** OpenTelemetry SDK → Datadog/Grafana. Metryki: latencja query,
koszt tokenów per request, retrieval hit rate. Alerty gdy koszt API przekroczy próg.

---

## 10. Słowniczek

| Termin | Definicja w kontekście projektu |
|---|---|
| **RAG** | Retrieval-Augmented Generation — architektura gdzie LLM odpowiada na pytania na podstawie dokumentów pobranych w czasie rzeczywistym, nie z pamięci modelu |
| **LLM** | Large Language Model — model językowy (tu: Claude) generujący tekst na podstawie promptu |
| **Embedding** | Reprezentacja tekstu jako wektor liczb (1536 float32) — semantycznie podobne teksty mają bliskie wektory |
| **Wektor** | Lista liczb zmiennoprzecinkowych reprezentująca tekst w przestrzeni wielowymiarowej |
| **Cosine similarity** | Miara podobieństwa dwóch wektorów (cosinus kąta) — 1.0 = identyczne, 0.0 = ortogonalne |
| **Chunking** | Podział długich dokumentów na mniejsze fragmenty do osobnego indeksowania |
| **Overlap** | Liczba słów wspólnych między sąsiednimi chunkami — zapobiega utracie informacji na granicach |
| **Prompt injection** | Atak gdzie użytkownik wstrzykuje instrukcje dla LLM w treści pytania |
| **Context window** | Maksymalna liczba tokenów widoczna przez LLM — prompt + dokumenty + pytanie |
| **Halucynacja** | LLM generuje przekonująco brzmiące, ale fałszywe informacje |
| **SSE** | Server-Sent Events — protokół HTTP do jednostronnego streamingu serwer → klient |
| **Streaming** | Wysyłanie odpowiedzi partiami w czasie rzeczywistym zamiast czekania na całość |
| **SOLID** | 5 zasad OOP: Single Responsibility, Open/Closed, Liskov, Interface Segregation, Dependency Inversion |
| **DRY** | Don't Repeat Yourself — każda logika w jednym miejscu, nie duplikowana |
| **Dependency Injection** | Wzorzec gdzie obiekt dostaje zależności z zewnątrz zamiast tworzyć je sam |
| **lifespan event** | FastAPI context manager wykonujący kod przy starcie i zatrzymaniu aplikacji |
| **CORS** | Cross-Origin Resource Sharing — mechanizm bezpieczeństwa przeglądarki blokujący cross-origin requesty |
| **Docker Compose** | Narzędzie do uruchamiania wielu kontenerów jako jeden serwis ze wspólną siecią |
| **Healthcheck** | Endpoint `/health` zwracający status aplikacji — używany przez load balancery i monitoring |
| **Conventional commits** | Standard formatowania: `feat:`, `fix:`, `chore:`, `docs:` — czytelna historia git |
| **Top-k retrieval** | Pobieranie k najbardziej podobnych chunków (tu: `TOP_K_CHUNKS = 10`) |
| **text-embedding-3-small** | Model OpenAI generujący embeddingi — wielojęzyczny, 1536 dim, dobry stosunek ceny do jakości |
| **claude-sonnet-4** | Model Anthropic użyty do generowania odpowiedzi w tym projekcie |
| **HNSW** | Hierarchical Navigable Small World — algorytm indeksowania wektorów, O(log N) search w pgvector |
| **IoC container** | Inversion of Control container — mechanizm (tu: `Depends()` w FastAPI) automatycznie rozwiązujący zależności |
