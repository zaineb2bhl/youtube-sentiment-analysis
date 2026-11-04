"""API de production conteneurisée (Docker) — déployée sur GitHub Codespaces, port 7860.

Version autonome et allégée de src/api : un seul fichier. Seule dépendance interne :
src/utils/text_cleaning.py, pour garantir exactement le même prétraitement qu'à l'entraînement.
Le port est configurable via la variable d'environnement PORT (défaut 7860).
"""
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from src.utils.text_cleaning import clean_text

MODELS_DIR = Path(os.getenv("MODELS_DIR", Path(__file__).resolve().parent / "models"))
MAX_BATCH = int(os.getenv("MAX_BATCH", 500))
MAX_CHARS = 5000
LABEL_TO_NAME = {-1: "negative", 0: "neutral", 1: "positive"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("sentiment-api")
ARTIFACTS: dict = {}


class BatchRequest(BaseModel):
    comments: List[str] = Field(..., min_length=1, max_length=MAX_BATCH)

    @field_validator("comments")
    @classmethod
    def check(cls, v):
        if all(not c.strip() for c in v):
            raise ValueError("tous les commentaires sont vides")
        return [c[:MAX_CHARS] for c in v]


class Prediction(BaseModel):
    text: str
    label: int
    sentiment: str
    confidence: float
    probabilities: Dict[str, float]


class Stats(BaseModel):
    total: int
    counts: Dict[str, int]
    percentages: Dict[str, float]
    average_confidence: float
    overall_sentiment: str


class BatchResponse(BaseModel):
    predictions: List[Prediction]
    stats: Stats
    processing_time_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        ARTIFACTS["vectorizer"] = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
        ARTIFACTS["model"] = joblib.load(MODELS_DIR / "sentiment_model.joblib")
        logger.info("Modèle chargé depuis %s", MODELS_DIR)
    except Exception:
        logger.exception("Impossible de charger le modèle")
    yield
    ARTIFACTS.clear()


app = FastAPI(title="YouTube Sentiment API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST", "OPTIONS"],
                   allow_headers=["*"])


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    logger.exception("Erreur sur %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Erreur interne du serveur"})


@app.get("/")
def root():
    return {"message": "YouTube Sentiment API", "docs": "/docs",
            "endpoints": ["/health", "/predict_batch"]}


@app.get("/health")
def health():
    loaded = "model" in ARTIFACTS
    return {"status": "ok" if loaded else "degraded", "model_loaded": loaded,
            "model_version": "1.0.0"}


@app.post("/predict_batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    if "model" not in ARTIFACTS:
        raise HTTPException(status_code=503, detail="Modèle non disponible")
    t0 = time.perf_counter()
    vec, model = ARTIFACTS["vectorizer"], ARTIFACTS["model"]
    cleaned = [clean_text(c) for c in req.comments]
    proba = model.predict_proba(vec.transform(cleaned))
    classes = [int(c) for c in model.classes_]

    preds, counts = [], {n: 0 for n in LABEL_TO_NAME.values()}
    for text, clean, p in zip(req.comments, cleaned, proba):
        if clean:
            b = int(p.argmax())
            label, conf = classes[b], float(p[b])
            probs = {LABEL_TO_NAME[c]: round(float(x), 4) for c, x in zip(classes, p)}
        else:
            label, conf = 0, 1.0
            probs = {LABEL_TO_NAME[c]: float(c == 0) for c in classes}
        name = LABEL_TO_NAME[label]
        counts[name] += 1
        preds.append({"text": text, "label": label, "sentiment": name,
                      "confidence": round(conf, 4), "probabilities": probs})

    n = len(preds)
    score = float(np.mean([p["label"] for p in preds]))
    stats = {
        "total": n,
        "counts": counts,
        "percentages": {k: round(100 * v / n, 2) for k, v in counts.items()},
        "average_confidence": round(float(np.mean([p["confidence"] for p in preds])), 4),
        "overall_sentiment": "positive" if score > 0.15 else "negative" if score < -0.15 else "neutral",
    }
    ms = round((time.perf_counter() - t0) * 1000, 2)
    logger.info("predict_batch n=%d time=%.1fms", n, ms)
    return {"predictions": preds, "stats": stats, "processing_time_ms": ms}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))
