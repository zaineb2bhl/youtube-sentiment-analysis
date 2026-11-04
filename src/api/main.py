"""Phase 4 - API REST FastAPI (développement local).

Lancer : uvicorn src.api.main:app --reload --port 8000
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.predictor import SentimentPredictor
from src.api.schemas import BatchRequest, BatchResponse, HealthResponse

LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s",
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler(LOG_DIR / "api.log", encoding="utf-8")])
logger = logging.getLogger("sentiment-api")

predictor = SentimentPredictor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Chargement du modèle UNE seule fois au démarrage (pas à chaque requête)
    try:
        predictor.load()
        logger.info("Modèle chargé (version %s)", predictor.version)
    except Exception:
        logger.exception("Échec du chargement du modèle")
    yield


app = FastAPI(title="YouTube Sentiment API", version="1.0.0",
              description="Analyse de sentiment de commentaires YouTube (TF-IDF + LogReg)",
              lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # l'extension Chrome a une origine chrome-extension://<id>
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    logger.exception("Erreur non gérée sur %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Erreur interne du serveur"})


@app.get("/")
def root():
    return {"message": "YouTube Sentiment API", "docs": "/docs",
            "endpoints": ["/health", "/predict_batch"]}


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok" if predictor.loaded else "degraded",
                          model_loaded=predictor.loaded, model_version=predictor.version)


@app.post("/predict_batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    if not predictor.loaded:
        raise HTTPException(status_code=503, detail="Modèle non disponible")
    result = predictor.predict(req.comments)
    logger.info("predict_batch n=%d time=%.1fms", len(req.comments), result["processing_time_ms"])
    return result
