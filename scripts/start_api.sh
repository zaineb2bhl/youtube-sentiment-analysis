#!/usr/bin/env bash
# Construit l'image et (re)lance le conteneur de l'API sur le port 7860.
# Exécuté automatiquement à chaque démarrage du Codespace (.devcontainer/devcontainer.json),
# utile après l'arrêt automatique pour inactivité.
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE=youtube-sentiment-api
NAME=sentiment-api

echo ">> attente du daemon Docker"
for i in $(seq 1 60); do
  docker info >/dev/null 2>&1 && break
  sleep 2
done
docker info >/dev/null 2>&1 || { echo "!! Docker indisponible" >&2; exit 1; }

echo ">> docker build"
docker build -t "$IMAGE" .

echo ">> docker run"
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --restart unless-stopped -p 7860:7860 "$IMAGE"

for i in $(seq 1 30); do
  if curl -sf http://localhost:7860/health >/dev/null; then
    echo ">> API prête : $(curl -s http://localhost:7860/health)"
    if [ -n "${CODESPACE_NAME:-}" ]; then
      echo ">> URL publique : https://${CODESPACE_NAME}-7860.${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN:-app.github.dev}"
      echo ">> Pensez à rendre le port 7860 PUBLIC (onglet Ports → clic droit → Port Visibility → Public)"
    fi
    exit 0
  fi
  sleep 1
done
echo "!! L'API ne répond pas, voir : docker logs $NAME" >&2
exit 1
