"""FastAPI app exposing the pipeline over HTTP."""

from fastapi import FastAPI

app = FastAPI(title="CheckTheBall", version="0.0.1")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "checktheball"}
