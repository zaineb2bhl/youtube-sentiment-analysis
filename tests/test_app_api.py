"""Tests de l'API de production (app_api.py, celle qui tourne dans le conteneur Docker)."""
import pytest
from fastapi.testclient import TestClient

import app_api


@pytest.fixture(scope="module")
def client():
    with TestClient(app_api.app) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] is True


def test_docs_available(client):
    assert client.get("/docs").status_code == 200


def test_predict_batch(client):
    r = client.post("/predict_batch", json={"comments": ["I love it", "horrible video", ""]})
    assert r.status_code == 200
    body = r.json()
    assert [p["sentiment"] for p in body["predictions"]][:2] == ["positive", "negative"]
    assert body["predictions"][2]["sentiment"] == "neutral"  # vide après nettoyage
    assert body["stats"]["total"] == 3


@pytest.mark.parametrize("payload", [{"comments": []}, {"comments": "x"}, {}])
def test_invalid(client, payload):
    assert client.post("/predict_batch", json=payload).status_code == 422


def test_same_results_as_dev_api(client):
    """Cohérence : l'API de prod et l'API de dev renvoient les mêmes prédictions."""
    from src.api.main import app as dev_app
    texts = ["great job", "worst ever", "what is this song", "😍🔥", "not bad at all"]
    with TestClient(dev_app) as dev:
        a = dev.post("/predict_batch", json={"comments": texts}).json()["predictions"]
    b = client.post("/predict_batch", json={"comments": texts}).json()["predictions"]
    assert [(p["label"], p["confidence"]) for p in a] == [(p["label"], p["confidence"]) for p in b]
