"""Chargement unique du modèle et logique de prédiction."""
import time
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np

from src.utils.text_cleaning import clean_text

LABEL_TO_NAME = {-1: "negative", 0: "neutral", 1: "positive"}
DEFAULT_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


class SentimentPredictor:
    def __init__(self, models_dir: Path = DEFAULT_MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.vectorizer = None
        self.model = None
        self.version = "unloaded"

    def load(self) -> None:
        self.vectorizer = joblib.load(self.models_dir / "tfidf_vectorizer.joblib")
        self.model = joblib.load(self.models_dir / "sentiment_model.joblib")
        mtime = (self.models_dir / "sentiment_model.joblib").stat().st_mtime
        self.version = time.strftime("%Y%m%d-%H%M", time.localtime(mtime))

    @property
    def loaded(self) -> bool:
        return self.model is not None and self.vectorizer is not None

    def predict(self, comments: List[str]) -> dict:
        if not self.loaded:
            raise RuntimeError("Modèle non chargé")
        t0 = time.perf_counter()
        cleaned = [clean_text(c) for c in comments]
        proba = self.model.predict_proba(self.vectorizer.transform(cleaned))
        classes = [int(c) for c in self.model.classes_]
        best = proba.argmax(axis=1)

        predictions = []
        for text, clean, p, b in zip(comments, cleaned, proba, best):
            if not clean:  # texte vide après nettoyage (ex. uniquement des symboles)
                label, conf = 0, 1.0
                probs = {LABEL_TO_NAME[c]: float(c == 0) for c in classes}
            else:
                label, conf = classes[b], float(p[b])
                probs = {LABEL_TO_NAME[c]: round(float(pi), 4) for c, pi in zip(classes, p)}
            predictions.append({"text": text, "label": label, "sentiment": LABEL_TO_NAME[label],
                                "confidence": round(conf, 4), "probabilities": probs})

        stats = self._stats(predictions)
        return {"predictions": predictions, "stats": stats,
                "processing_time_ms": round((time.perf_counter() - t0) * 1000, 2)}

    @staticmethod
    def _stats(predictions: List[dict]) -> dict:
        n = len(predictions)
        counts = {name: 0 for name in LABEL_TO_NAME.values()}
        for p in predictions:
            counts[p["sentiment"]] += 1
        score = np.mean([p["label"] for p in predictions])
        overall = "positive" if score > 0.15 else "negative" if score < -0.15 else "neutral"
        return {
            "total": n,
            "counts": counts,
            "percentages": {k: round(100 * v / n, 2) for k, v in counts.items()},
            "average_confidence": round(float(np.mean([p["confidence"] for p in predictions])), 4),
            "overall_sentiment": overall,
        }


predictor: Optional[SentimentPredictor] = None
