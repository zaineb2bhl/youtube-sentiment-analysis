// URLs de l'API. PROD_API_URL = adresse publique du port 7860 du Codespace GitHub
// (https://<nom-du-codespace>-7860.app.github.dev). Elle change si on recrée un Codespace.
const CONFIG = {
  PROD_API_URL: "https://VOTRE-CODESPACE-7860.app.github.dev",
  LOCAL_API_URL: "http://127.0.0.1:8000",
  BATCH_SIZE: 100,          // nombre de commentaires envoyés par requête
  REQUEST_TIMEOUT_MS: 30000 // marge pour un Codespace qui vient de démarrer
};
