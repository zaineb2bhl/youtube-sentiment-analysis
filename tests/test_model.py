import json
import time
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score

from src.utils.text_cleaning import clean_text

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
TEST_CSV = ROOT / "data" / "processed" / "test.csv"

pytestmark = pytest.mark.skipif(not (MODELS / "sentiment_model.joblib").exists(),
                                reason="modèle non entraîné")


@pytest.fixture(scope="module")
def artifacts():
    return (joblib.load(MODELS / "tfidf_vectorizer.joblib"),
            joblib.load(MODELS / "sentiment_model.joblib"))


@pytest.mark.skipif(not TEST_CSV.exists(), reason="test set absent")
def test_performance_on_test_set(artifacts):
    vec, model = artifacts
    df = pd.read_csv(TEST_CSV).dropna()
    pred = model.predict(vec.transform(df["text"]))
    assert accuracy_score(df["label"], pred) >= 0.80
    assert min(f1_score(df["label"], pred, average=None)) > 0.75


@pytest.mark.parametrize("text", [
    "a", "", "!!!", "ok " * 2000, "😍🔥👍", "C'est génial, really amazing!", "这个视频很好",
])
def test_edge_cases_do_not_crash(artifacts, text):
    vec, model = artifacts
    proba = model.predict_proba(vec.transform([clean_text(text)]))
    assert proba.shape == (1, 3)
    assert abs(proba.sum() - 1) < 1e-6


def test_obvious_sentiments(artifacts):
    vec, model = artifacts
    texts = ["i love this video it is amazing", "this is horrible and stupid"]
    assert list(model.predict(vec.transform(texts))) == [1, -1]


def test_inference_time_batch_50(artifacts):
    vec, model = artifacts
    batch = [clean_text("this video is really great, thanks for sharing")] * 50
    model.predict_proba(vec.transform(batch))  # warm-up
    t0 = time.perf_counter()
    model.predict_proba(vec.transform(batch))
    assert (time.perf_counter() - t0) * 1000 < 100
