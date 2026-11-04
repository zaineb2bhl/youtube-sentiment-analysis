"""Schémas Pydantic de validation des requêtes/réponses."""
from typing import Dict, List

from pydantic import BaseModel, Field, field_validator

MAX_BATCH = 500
MAX_CHARS = 5000


class BatchRequest(BaseModel):
    comments: List[str] = Field(..., min_length=1, max_length=MAX_BATCH,
                                examples=[["I love this video!", "This is terrible", "ok"]])

    @field_validator("comments")
    @classmethod
    def check_comments(cls, v: List[str]) -> List[str]:
        if all(not c.strip() for c in v):
            raise ValueError("tous les commentaires sont vides")
        return [c[:MAX_CHARS] for c in v]


class Prediction(BaseModel):
    text: str
    label: int = Field(..., description="-1 négatif, 0 neutre, 1 positif")
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


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
