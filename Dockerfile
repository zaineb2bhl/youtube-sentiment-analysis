# Image légère. Python 3.11 (et non 3.10) : scikit-learn 1.8, utilisé pour entraîner le modèle,
# exige Python >= 3.11, et un modèle joblib doit être rechargé avec la même version de sklearn.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=7860

# Utilisateur non-root
RUN useradd -m -u 1000 appuser
WORKDIR /app

# 1) Dépendances d'abord : couche mise en cache tant que requirements-prod.txt ne change pas
COPY requirements-prod.txt .
RUN pip install -r requirements-prod.txt

# 2) Uniquement les fichiers nécessaires à l'exécution (le code change plus souvent → en dernier)
COPY src/__init__.py src/__init__.py
COPY src/utils/__init__.py src/utils/text_cleaning.py src/utils/
COPY models/tfidf_vectorizer.joblib models/sentiment_model.joblib models/
COPY app_api.py .

RUN chown -R appuser /app
USER appuser
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/health')" || exit 1

# Forme shell pour que ${PORT} soit interprété (port configurable : docker run -e PORT=8080 ...)
CMD uvicorn app_api:app --host 0.0.0.0 --port ${PORT}
