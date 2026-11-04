# YouTube Sentiment Analyzer — Projet MLOps complet

> TP3 Cloud Computing — ENSAM Rabat, filière INDIA (2026/27)

Système complet qui analyse automatiquement le sentiment (positif / neutre / négatif) des
commentaires d'une vidéo YouTube : un modèle scikit-learn est servi par une API FastAPI
conteneurisée avec Docker et exécutée dans le cloud sur GitHub Codespaces (URL publique HTTPS), et une extension Chrome sert d'interface.

## Architecture

```
┌──────────────────────┐   batch JSON    ┌───────────────────────────┐
│  Extension Chrome    │ ──────────────▶ │  API FastAPI (Docker)     │
│  content.js : extrait│                 │  GitHub Codespaces :7860  │
│  les commentaires    │ ◀────────────── │  /health  /predict_batch  │
│  popup : stats, liste│  sentiments +   │  clean_text → TF-IDF → LR │
│  filtres, export     │  statistiques   └───────────────────────────┘
└──────────────────────┘                          ▲ models/*.joblib
                                                  │
        data/raw ─▶ preprocess ─▶ data/processed ─▶ train (GridSearchCV)
```

| Composant | Technologie |
|---|---|
| Frontend | Extension Chrome Manifest V3 (JavaScript, HTML, CSS) |
| Backend | FastAPI + Pydantic, Uvicorn |
| ML | TF-IDF + Logistic Regression (scikit-learn) |
| Déploiement | Docker dans GitHub Codespaces (port 7860 public) |
| Versioning | Git + GitHub |

## Structure du projet

```
├── data/
│   ├── raw/                  # dataset brut (reddit.csv, non versionné)
│   └── processed/            # données nettoyées + train/test (non versionnées)
├── models/                   # tfidf_vectorizer.joblib, sentiment_model.joblib
├── reports/                  # metrics.json + figures (EDA, matrice de confusion, comparaison)
├── src/
│   ├── data/                 # download_data.py, preprocess.py
│   ├── models/               # train.py
│   ├── api/                  # main.py, predictor.py, schemas.py (API de développement)
│   └── utils/                # text_cleaning.py (partagé entraînement / API)
├── chrome-extension/         # manifest.json, content.js, popup.html/css/js, config.js
├── app_api.py                # API de production (exécutée dans le conteneur)
├── Dockerfile, .dockerignore # image Docker de l'API
├── requirements-prod.txt     # dépendances de production uniquement (versions figées)
├── .devcontainer/            # config Codespaces : Docker + build/lancement auto de l'API
├── scripts/                  # start_api.sh (build + run Docker), check_live_api.py
├── tests/                    # tests pytest (nettoyage, modèle, API)
└── logs/                     # logs de l'API
```

## Installation

```bash
python -m venv .venv
.\.venv\Scripts\activate          # Windows PowerShell
# source .venv/bin/activate       # Linux / macOS
pip install -r requirements.txt
```

## Pipeline ML

```bash
python src/data/download_data.py    # télécharge data/raw/reddit.csv + statistiques
python src/data/preprocess.py       # nettoyage, EDA, split 80/20 stratifié (random_state=42)
python src/models/train.py          # comparaison, GridSearchCV, évaluation, sauvegarde joblib
```

### Données

Dataset Reddit Sentiment Analysis (37 249 commentaires → 36 555 après suppression des NaN,
des textes vides et des doublons).

| Classe | Nombre | Part |
|---|---|---|
| -1 négatif | 8 233 | 22,5 % |
| 0 neutre | 12 600 | 34,5 % |
| 1 positif | 15 722 | 43,0 % |

Nettoyage : minuscules, emojis courants convertis en mots (😍 → *love*, 😭 → *sad*),
suppression des URLs, des mentions/hashtags, du HTML et des caractères spéciaux, réduction
des lettres répétées. Les négations (*not*, *don't*) sont conservées.
Déséquilibre (ratio 1,9) géré par `class_weight="balanced"`.

### Résultats (test set, 7 311 commentaires)

| Modèle (TF-IDF commun) | Accuracy | F1-macro |
|---|---|---|
| **Logistic Regression optimisée (retenue)** | **0,894** | **0,885** |
| LinearSVC | 0,881 | 0,870 |
| Logistic Regression (C=1) | 0,848 | 0,838 |
| Random Forest | 0,768 | 0,706 |
| Complement Naive Bayes | 0,665 | 0,624 |

Meilleurs hyperparamètres (GridSearchCV, CV 3 folds, F1-macro) : `C=10`, unigrammes,
`max_features=30000`, `sublinear_tf=True`.

| Classe | Précision | Rappel | F1 |
|---|---|---|---|
| Négatif | 0,842 | 0,811 | 0,826 |
| Neutre | 0,895 | 0,967 | 0,930 |
| Positif | 0,920 | 0,879 | 0,899 |

| Critère du TP | Objectif | Obtenu |
|---|---|---|
| Accuracy | ≥ 80 % | ✅ 89,4 % |
| F1 par classe | > 0,75 | ✅ min 0,826 |
| Inférence batch de 50 | < 100 ms | ✅ ~4 ms |

Figures : `reports/figures/eda_distribution.png`, `confusion_matrix.png`, `model_comparison.png`.

## API

### Lancer en local

```bash
uvicorn src.api.main:app --reload --port 8000
# Documentation interactive : http://127.0.0.1:8000/docs
```

### Endpoints

| Endpoint | Méthode | Description |
|---|---|---|
| `/health` | GET | État de l'API et du modèle |
| `/predict_batch` | POST | Analyse un batch (1 à 500 commentaires) → sentiments + statistiques |

```bash
curl -X POST http://127.0.0.1:8000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{"comments": ["I love this video!", "This is terrible", "What camera is that?"]}'
```

```json
{
  "predictions": [
    {"text": "I love this video!", "label": 1, "sentiment": "positive", "confidence": 0.99,
     "probabilities": {"negative": 0.0, "neutral": 0.01, "positive": 0.99}}
  ],
  "stats": {"total": 3, "counts": {"negative": 1, "neutral": 1, "positive": 1},
            "percentages": {"negative": 33.33, "neutral": 33.33, "positive": 33.33},
            "average_confidence": 0.98, "overall_sentiment": "neutral"},
  "processing_time_ms": 2.7
}
```

Erreurs : `422` pour une entrée invalide (batch vide, uniquement des textes vides, mauvais type,
plus de 500 commentaires), `503` si le modèle n'est pas chargé, `500` pour une erreur interne
(journalisée dans `logs/api.log`).

## Déploiement cloud : Docker sur GitHub Codespaces

> Hugging Face Spaces exige désormais un abonnement PRO pour les Spaces Docker. On utilise donc
> GitHub Codespaces, une VM cloud gratuite dans la limite du quota mensuel. Le Dockerfile reste le même.

**Fichiers concernés** : [app_api.py](app_api.py), [Dockerfile](Dockerfile), [.dockerignore](.dockerignore),
[requirements-prod.txt](requirements-prod.txt), [.devcontainer/devcontainer.json](.devcontainer/devcontainer.json),
[scripts/start_api.sh](scripts/start_api.sh).

1. Pousser le dépôt sur GitHub (code + Dockerfile + `models/*.joblib`).
2. Sur la page du dépôt : **Code → Codespaces → Create codespace on main**.
3. Au démarrage, `scripts/start_api.sh` construit l'image et lance le conteneur automatiquement.
   Pour le faire à la main :
   ```bash
   docker build -t youtube-sentiment-api .
   docker run -d -p 7860:7860 youtube-sentiment-api
   ```
4. Onglet **Ports** : clic droit sur 7860 → **Port Visibility → Public**.
5. Copier l'adresse `https://<nom>-7860.app.github.dev`, ouvrir `/docs` pour vérifier, puis tester :
   ```bash
   python scripts/check_live_api.py https://<nom>-7860.app.github.dev
   ```
6. Mettre cette URL dans `PROD_API_URL` de [chrome-extension/config.js](chrome-extension/config.js),
   recharger l'extension et choisir « Production » dans ⚙.

⚠️ L'URL ne fonctionne que tant que le Codespace est allumé (il s'arrête après environ 30 min d'inactivité).
Pour une démo, relancer le Codespace (le conteneur redémarre tout seul) et vérifier que le port 7860
est toujours **public**.

Bonnes pratiques appliquées :
- image `python:3.11-slim`. Python 3.11 et non 3.10, car scikit-learn 1.8, avec lequel le modèle a été entraîné, l'exige ;
- `.dockerignore` et `COPY` ciblés : seuls `app_api.py`, `text_cleaning.py` et les deux fichiers `.joblib` entrent dans l'image ;
- les dépendances sont installées avant la copie du code, pour profiter du cache Docker ;
- `PIP_NO_CACHE_DIR`, utilisateur non-root et `HEALTHCHECK` ;
- port configurable : `docker run -e PORT=8080 -p 8080:8080 youtube-sentiment-api`.

## Extension Chrome

### Installation
1. Ouvrir `chrome://extensions`, activer le **Mode développeur**.
2. Cliquer **Charger l'extension non empaquetée** et sélectionner le dossier `chrome-extension/`.
3. Épingler l'extension dans la barre d'outils.

### Utilisation
1. Ouvrir une vidéo YouTube.
2. Cliquer sur l'icône, puis **Analyser les commentaires**.
3. L'extension fait défiler la page pour charger les commentaires (jusqu'au maximum choisi),
   les envoie par batchs de 100 à l'API et affiche les résultats.

### Fonctionnalités
- Extraction automatique des commentaires (avec défilement pour le chargement dynamique), et réponses en option
- Statistiques globales : graphique en anneau, pourcentages, sentiment global, temps de réponse
- Liste des commentaires avec sentiment et barre de confiance
- Filtres positif / neutre / négatif, recherche, tri par confiance
- Mode sombre / clair (mémorisé)
- Copier les résultats, export CSV / JSON
- Indicateur d'état de l'API, choix local / production / URL personnalisée
- Gestion des erreurs (page non YouTube, aucun commentaire, API injoignable, délai dépassé)

## Tests

```bash
pytest -q                                    # 33 tests : nettoyage, modèle, API dev et prod
python scripts/check_live_api.py             # intégration contre l'API locale
python scripts/check_live_api.py <URL>       # intégration contre l'API Codespaces
```

## Notes MLOps

- Les données brutes et les données traitées sont séparées : `raw` reste immuable, ce qui rend le pipeline reproductible.
- Le même `clean_text` sert à l'entraînement et à l'inférence, ce qui évite les écarts de prétraitement entre les deux.
- Le modèle est chargé une seule fois au démarrage de l'API (`lifespan`), pas à chaque requête.
- Les versions de production sont figées (`deployment/requirements.txt`) : un fichier joblib doit être rechargé avec la même version de scikit-learn.
- L'image Docker est construite pour profiter du cache : les dépendances sont installées avant la copie du code, et le conteneur tourne avec un utilisateur non-root.
