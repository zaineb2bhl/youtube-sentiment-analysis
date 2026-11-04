"""Phase 7 - Validation complète : modèle, API locale, API cloud, cohérence.

Usage :
    python scripts/run_validation.py [URL_API_CLOUD]

Pré-requis : API locale lancée (uvicorn src.api.main:app --port 8000) et, pour la partie
cloud, Codespace allumé avec le port 7860 public.
Le rapport est enregistré dans reports/validation_results.md.
"""
import statistics
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import accuracy_score, classification_report, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.text_cleaning import clean_text  # noqa: E402

LOCAL = "http://127.0.0.1:8000"
CLOUD = (sys.argv[1] if len(sys.argv) > 1
         else "https://animated-trout-wr44q69xw5r529g44-7860.app.github.dev").rstrip("/")
NAMES = {-1: "negative", 0: "neutral", 1: "positive"}
OUT = []
RESULTS = []  # (catégorie, test, ok)


def log(line=""):
    print(line)
    OUT.append(line)


def check(cat, name, ok, detail=""):
    RESULTS.append((cat, name, ok))
    log(f"  [{'OK ' if ok else 'ECHEC'}] {name}{'  — ' + detail if detail else ''}")


def section(title):
    log(f"\n## {title}\n")


# --------------------------------------------------------------------------- modèle
def test_model():
    section("1. Tests du modèle")
    vec = joblib.load(ROOT / "models" / "tfidf_vectorizer.joblib")
    model = joblib.load(ROOT / "models" / "sentiment_model.joblib")

    log("### 1.1 Performance sur le test set\n")
    test = pd.read_csv(ROOT / "data" / "processed" / "test.csv").dropna()
    pred = model.predict(vec.transform(test["text"]))
    acc = accuracy_score(test["label"], pred)
    f1s = f1_score(test["label"], pred, average=None, labels=[-1, 0, 1])
    log("```")
    log(classification_report(test["label"], pred, labels=[-1, 0, 1],
                              target_names=["négatif", "neutre", "positif"], digits=4))
    log("```")
    check("Modèle", f"Accuracy ≥ 80 % ({len(test)} commentaires)", acc >= 0.80, f"{acc:.2%}")
    check("Modèle", "F1 par classe > 0,75", f1s.min() > 0.75,
          " / ".join(f"{n}={f:.3f}" for n, f in zip(["nég", "neu", "pos"], f1s)))

    log("\n### 1.2 Cas limites\n")
    cases = [
        ("Très court (1 lettre)", "a"),
        ("Très court (1 mot positif)", "great"),
        ("Très court (1 mot négatif)", "awful"),
        ("Vide", ""),
        ("Ponctuation seule", "!!!???"),
        ("Très long (~6 000 car.)", "this video is really amazing and helpful " * 150),
        ("Emojis seuls positifs", "😍🔥👍"),
        ("Emojis seuls négatifs", "😡👎"),
        ("Texte + emoji", "thanks for this tutorial 😊"),
        ("Majuscules / répétitions", "SOOOOO GOOOOD!!!!"),
        ("URL + mention", "@john check https://youtu.be/xyz this is horrible"),
        ("Mélange FR/EN", "C'est génial, really amazing work!"),
        ("Français seul", "Cette vidéo est nulle, je déteste"),
        ("Arabe", "فيديو رائع جدا"),
        ("Chinois", "这个视频很好"),
        ("Négation", "this is not good at all"),
        ("Sarcasme", "oh great, another useless video"),
    ]
    log("| Cas | Texte (extrait) | Prédiction | Confiance |")
    log("|---|---|---|---|")
    crashes = 0
    for label, text in cases:
        try:
            cleaned = clean_text(text)
            p = model.predict_proba(vec.transform([cleaned]))[0]
            if not cleaned:
                res, conf = "neutral (texte vide)", 1.0
            else:
                res, conf = NAMES[int(model.classes_[p.argmax()])], p.max()
            short = (text[:40] + "…") if len(text) > 40 else text
            log(f"| {label} | `{short}` | {res} | {conf:.2f} |")
        except Exception as e:  # noqa: BLE001
            crashes += 1
            log(f"| {label} | — | ERREUR {e} | — |")
    log("")
    check("Modèle", f"Aucun plantage sur {len(cases)} cas limites", crashes == 0)
    log("\n  Remarque : le modèle est entraîné sur de l'anglais. Le français, l'arabe ou le chinois sont")
    log("  donc surtout classés « neutre » (mots inconnus). La négation (« not good ») et le sarcasme sont mal gérés :")
    log("  le modèle retenu utilise des unigrammes, il voit « good » sans « not ». Piste : bigrammes ou modèle Transformer.")
    log("  Ce sont des limites connues de l'approche TF-IDF.")

    log("\n### 1.3 Temps d'inférence (nettoyage + TF-IDF + prédiction)\n")
    pool = test["text"].tolist()
    log("| Taille du batch | Médiane (ms) | P95 (ms) |")
    log("|---|---|---|")
    med50 = None
    for size in (1, 10, 50, 100, 500):
        batch = pool[:size]
        times = []
        for _ in range(30):
            t0 = time.perf_counter()
            model.predict_proba(vec.transform([clean_text(t) for t in batch]))
            times.append((time.perf_counter() - t0) * 1000)
        med, p95 = statistics.median(times), float(np.percentile(times, 95))
        if size == 50:
            med50 = med
        log(f"| {size} | {med:.2f} | {p95:.2f} |")
    log("")
    check("Modèle", "Inférence batch de 50 < 100 ms", med50 < 100, f"{med50:.2f} ms")


# --------------------------------------------------------------------------- API
def test_api(base, cat):
    try:
        r = requests.get(f"{base}/health", timeout=30)
    except requests.RequestException as e:
        check(cat, f"/health joignable ({base})", False, str(e)[:80])
        return None
    try:
        body = r.json()
    except ValueError:
        body = {}
    if not (r.ok and body.get("model_loaded")):
        check(cat, f"GET /health → 200 et modèle chargé ({base})", False,
              f"code {r.status_code} — Codespace arrêté ou port 7860 non public ?")
        return None
    check(cat, "GET /health → 200 et modèle chargé", True, str(body))
    check(cat, "GET /docs → 200", requests.get(f"{base}/docs", timeout=30).ok)

    pre = requests.options(f"{base}/predict_batch", timeout=30, headers={
        "Origin": "chrome-extension://test", "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type"})
    check(cat, "CORS autorisé pour l'extension Chrome",
          pre.headers.get("access-control-allow-origin") in ("*", "chrome-extension://test"))

    log(f"\n  Temps de réponse ({cat}) :\n")
    log("| Batch | Réseau + API, médiane (ms) | Temps modèle côté serveur (ms) | Statut |")
    log("|---|---|---|---|")
    sample = ["I love this video!", "worst tutorial ever", "what camera is this?",
              "thanks a lot 😊", "not good"]
    all_ok = True
    for size in (1, 10, 50, 100, 500):
        comments = (sample * (size // len(sample) + 1))[:size]
        times, srv = [], None
        for _ in range(5):
            t0 = time.perf_counter()
            r = requests.post(f"{base}/predict_batch", json={"comments": comments}, timeout=60)
            times.append((time.perf_counter() - t0) * 1000)
            ok = r.ok and len(r.json()["predictions"]) == size
            all_ok &= ok
            srv = r.json().get("processing_time_ms") if r.ok else None
        log(f"| {size} | {statistics.median(times):.1f} | {srv} | {'OK' if ok else 'ECHEC'} |")
    log("")
    check(cat, "POST /predict_batch, batchs de 1 à 500", all_ok)

    invalid = [
        ("batch vide", {"comments": []}),
        ("que des textes vides", {"comments": ["", "   "]}),
        ("mauvais type", {"comments": "pas une liste"}),
        ("mauvais champ", {"text": ["hello"]}),
        ("corps vide", {}),
        ("batch > 500", {"comments": ["x"] * 501}),
        ("élément non-texte", {"comments": [123, None]}),
    ]
    for name, payload in invalid:
        code = requests.post(f"{base}/predict_batch", json=payload, timeout=30).status_code
        check(cat, f"Erreur gérée : {name} → 422", code == 422, f"code {code}")
    code = requests.post(f"{base}/predict_batch", data="not json",
                         headers={"Content-Type": "application/json"}, timeout=30).status_code
    check(cat, "Erreur gérée : JSON malformé → 422", code == 422, f"code {code}")

    probe = ["I love this video, great explanation!", "This is the worst tutorial ever",
             "What camera do you use?", "😍🔥", "meh", "not bad at all", "I hate it"]
    return requests.post(f"{base}/predict_batch", json={"comments": probe}, timeout=30).json()


def main():
    log("# Rapport de validation — YouTube Sentiment Analyzer")
    log(f"\nDate : {time.strftime('%d/%m/%Y %H:%M')}  ·  API locale : {LOCAL}  ·  API cloud : {CLOUD}")
    test_model()

    section("2. Tests de l'API locale (uvicorn, src/api/main.py)")
    local = test_api(LOCAL, "API locale")
    section("3. Tests de l'API déployée (Docker sur GitHub Codespaces, app_api.py)")
    cloud = test_api(CLOUD, "API cloud")

    section("4. Intégration : cohérence entre API locale et API cloud")
    if local and cloud:
        log("| Commentaire | Locale | Cloud |")
        log("|---|---|---|")
        same = True
        for a, b in zip(local["predictions"], cloud["predictions"]):
            eq = a["label"] == b["label"] and abs(a["confidence"] - b["confidence"]) < 1e-6
            same &= eq
            log(f"| {a['text']} | {a['sentiment']} ({a['confidence']:.2f}) | "
                f"{b['sentiment']} ({b['confidence']:.2f}) |")
        log("")
        check("Intégration", "Prédictions identiques en local et en cloud", same)
        check("Intégration", "Statistiques identiques",
              local["stats"]["counts"] == cloud["stats"]["counts"])
    else:
        check("Intégration", "Comparaison local / cloud", False, "une des deux API est injoignable")

    section("Synthèse")
    log("| Catégorie | Tests réussis |")
    log("|---|---|")
    for cat in dict.fromkeys(c for c, _, _ in RESULTS):
        items = [ok for c, _, ok in RESULTS if c == cat]
        log(f"| {cat} | {sum(items)}/{len(items)} |")
    total = sum(ok for _, _, ok in RESULTS)
    log(f"| **Total** | **{total}/{len(RESULTS)}** |")

    out = ROOT / "reports" / "validation_results.md"
    out.write_text("\n".join(OUT) + "\n", encoding="utf-8")
    print(f"\nRapport enregistré : {out}")
    sys.exit(0 if total == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
