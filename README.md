# Company KB RAG

A company internal knowledge base chatbot powered by Retrieval-Augmented Generation (RAG).
Users ask questions in natural language; answers are grounded exclusively in company documents.

## Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI, Python 3.11, Uvicorn |
| LLM | Anthropic Claude (claude-sonnet-4-20250514) |
| Embeddings | OpenAI text-embedding-3-small |
| Similarity | NumPy cosine similarity |
| Frontend | Next.js 14 App Router, TypeScript strict, Tailwind CSS |
| Auth | NextAuth.js (configured in Step 8) |
| DevOps | GitHub Actions CI, Docker Compose |

## Setup (placeholder — detailed instructions in feat/etap-7-docker)

```bash
# Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp ../.env.example .env   # fill in API keys
uvicorn main:app --reload --port 8000

# Frontend
cd frontend
npm install
cp ../.env.example .env.local
npm run dev   # runs on port 3001
```

## CI Pipeline

Every pull request to `main` runs six jobs:

| Job | Tool | What it checks |
|---|---|---|
| `backend-lint` | flake8 | PEP 8 style, unused imports, line length |
| `backend-typecheck` | mypy --strict | Full static type safety across all Python modules |
| `backend-test` | pytest + pytest-cov | Unit tests pass, coverage report printed |
| `frontend-typecheck` | tsc --noEmit | TypeScript strict mode — zero type errors |
| `frontend-lint` | ESLint | Next.js + TypeScript lint rules, zero warnings |
| `frontend-build` | next build | Production build succeeds (catches missing env vars, broken imports) |

## Architecture

```
Browser
  └─► Next.js 14 (port 3000/3001)
        └─► FastAPI (port 8000)
              ├─► VectorStore (in-memory NumPy)
              ├─► OpenAI Embeddings API
              └─► Anthropic Claude API
```

## Project Status

- [ ] Step 1 — repo structure + CI pipeline
- [ ] Step 2 — ingestion pipeline (chunker, embeddings, vector store)
- [ ] Step 3 — query pipeline (RAG + summarization endpoint)
- [ ] Step 4 — Next.js chat UI
- [ ] Step 5 — company documents + auto-ingestion
- [ ] Step 6 — streaming + error handling + loading states
- [ ] Step 7 — Docker Compose + full documentation
