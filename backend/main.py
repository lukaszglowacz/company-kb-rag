from fastapi import FastAPI

app = FastAPI(title="Company KB RAG API")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
