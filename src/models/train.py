"""Phase 3 - Entraînement, comparaison, optimisation et sauvegarde du modèle."""
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
RANDOM_STATE = 42
LABELS = [-1, 0, 1]
NAMES = ["négatif", "neutre", "positif"]


def tfidf(**kw) -> TfidfVectorizer:
    params = dict(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True,
                  max_features=50_000)
    params.update(kw)
    return TfidfVectorizer(**params)


def compare_models(X_tr, y_tr, X_te, y_te) -> pd.DataFrame:
    """Compare plusieurs algorithmes avec un TF-IDF commun."""
    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "LinearSVC": LinearSVC(class_weight="balanced"),
        "ComplementNB": ComplementNB(),
        "RandomForest": RandomForestClassifier(n_estimators=150, n_jobs=-1,
                                               class_weight="balanced",
                                               random_state=RANDOM_STATE),
    }
    rows = []
    for name, clf in candidates.items():
        pipe = Pipeline([("tfidf", tfidf()), ("clf", clf)])
        t0 = time.perf_counter()
        pipe.fit(X_tr, y_tr)
        fit_s = time.perf_counter() - t0
        pred = pipe.predict(X_te)
        rows.append({"model": name, "accuracy": accuracy_score(y_te, pred),
                     "f1_macro": f1_score(y_te, pred, average="macro"), "fit_s": fit_s})
        print(f"  {name:20s} acc={rows[-1]['accuracy']:.4f} f1={rows[-1]['f1_macro']:.4f} "
              f"({fit_s:.1f}s)")
    return pd.DataFrame(rows).sort_values("f1_macro", ascending=False)


def tune_logreg(X_tr, y_tr) -> GridSearchCV:
    pipe = Pipeline([("tfidf", tfidf()),
                     ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))])
    grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__max_features": [30_000, 50_000, None],
        "clf__C": [1, 5, 10, 20],
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(pipe, grid, scoring="f1_macro", cv=cv, n_jobs=-1, verbose=1)
    search.fit(X_tr, y_tr)
    print(f"  Meilleurs paramètres : {search.best_params_}")
    print(f"  F1-macro CV : {search.best_score_:.4f}")
    return search


def evaluate(pipe: Pipeline, X_te, y_te, comparison: pd.DataFrame, best_params: dict) -> dict:
    pred = pipe.predict(X_te)
    report = classification_report(y_te, pred, labels=LABELS, target_names=NAMES,
                                   output_dict=True, digits=4)
    print(classification_report(y_te, pred, labels=LABELS, target_names=NAMES, digits=4))

    # Temps d'inférence : batch de 50 commentaires (vectorisation + prédiction)
    batch = list(X_te[:50])
    timings = []
    for _ in range(30):
        t0 = time.perf_counter()
        pipe.predict_proba(batch)
        timings.append((time.perf_counter() - t0) * 1000)
    infer_ms = float(np.median(timings))
    print(f"Inférence batch de 50 : {infer_ms:.1f} ms (médiane sur 30 essais)")

    cm = confusion_matrix(y_te, pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=NAMES).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Matrice de confusion (test)")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "figures" / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    comp = comparison.sort_values("f1_macro")
    ax.barh(comp["model"], comp["f1_macro"], color="#3b6fd8")
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1-macro (test)")
    ax.set_title("Comparaison des algorithmes (TF-IDF commun)")
    for i, v in enumerate(comp["f1_macro"]):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "figures" / "model_comparison.png", dpi=120)
    plt.close(fig)

    return {
        "accuracy": report["accuracy"],
        "f1_macro": report["macro avg"]["f1-score"],
        "per_class": {n: {k: report[n][k] for k in ("precision", "recall", "f1-score")}
                      for n in NAMES},
        "inference_ms_batch50": infer_ms,
        "confusion_matrix": cm.tolist(),
        "best_params": {k: (list(v) if isinstance(v, tuple) else v)
                        for k, v in best_params.items()},
        "comparison": comparison.to_dict(orient="records"),
        "criteria": {
            "accuracy>=0.80": report["accuracy"] >= 0.80,
            "f1_per_class>0.75": all(report[n]["f1-score"] > 0.75 for n in NAMES),
            "inference<100ms": infer_ms < 100,
        },
    }


def main():
    (REPORTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    train = pd.read_csv(PROCESSED_DIR / "train.csv").dropna()
    test = pd.read_csv(PROCESSED_DIR / "test.csv").dropna()
    X_tr, y_tr = train["text"].values, train["label"].values
    X_te, y_te = test["text"].values, test["label"].values

    print("[1/3] Comparaison des algorithmes")
    comparison = compare_models(X_tr, y_tr, X_te, y_te)
    print("\n[2/3] GridSearchCV sur TF-IDF + Logistic Regression")
    search = tune_logreg(X_tr, y_tr)
    best = search.best_estimator_
    print("\n[3/3] Évaluation sur le test set")
    metrics = evaluate(best, X_te, y_te, comparison, search.best_params_)

    joblib.dump(best.named_steps["tfidf"], MODELS_DIR / "tfidf_vectorizer.joblib", compress=3)
    joblib.dump(best.named_steps["clf"], MODELS_DIR / "sentiment_model.joblib", compress=3)
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False),
                                              encoding="utf-8")
    print(f"\nModèle et vectoriseur sauvegardés dans {MODELS_DIR}")
    print(f"Critères : {metrics['criteria']}")


if __name__ == "__main__":
    main()
