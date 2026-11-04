import time

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


@pytest.mark.parametrize("size", [1, 10, 50, 200])
def test_predict_batch_sizes(client, size):
    r = client.post("/predict_batch", json={"comments": ["great video"] * size})
    assert r.status_code == 200
    body = r.json()
    assert len(body["predictions"]) == size
    assert body["stats"]["total"] == size
    assert abs(sum(body["stats"]["percentages"].values()) - 100) < 0.1


def test_prediction_fields(client):
    p = client.post("/predict_batch", json={"comments": ["I love it"]}).json()["predictions"][0]
    assert set(p) == {"text", "label", "sentiment", "confidence", "probabilities"}
    assert p["label"] in (-1, 0, 1)
    assert 0 <= p["confidence"] <= 1


@pytest.mark.parametrize("payload", [
    {"comments": []},            # batch vide
    {"comments": ["   ", ""]},   # que des vides
    {"comments": "not a list"},  # mauvais type
    {"text": ["hello"]},         # mauvais champ
    {"comments": ["x"] * 501},   # batch trop grand
])
def test_invalid_payloads(client, payload):
    assert client.post("/predict_batch", json=payload).status_code == 422


def test_response_time(client):
    t0 = time.perf_counter()
    r = client.post("/predict_batch", json={"comments": ["nice explanation thanks"] * 50})
    assert r.status_code == 200
    assert (time.perf_counter() - t0) * 1000 < 500
