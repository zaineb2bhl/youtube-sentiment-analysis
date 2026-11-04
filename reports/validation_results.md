# Rapport de validation — YouTube Sentiment Analyzer

Date : 04/11/2026 12:06  ·  API locale : http://127.0.0.1:8000  ·  API cloud : https://animated-trout-wr44q69xw5r529g44-7860.app.github.dev

## 1. Tests du modèle

### 1.1 Performance sur le test set

```
              precision    recall  f1-score   support

     négatif     0.8417    0.8106    0.8259      1647
      neutre     0.8953    0.9671    0.9298      2520
     positif     0.9201    0.8788    0.8990      3144

    accuracy                         0.8939      7311
   macro avg     0.8857    0.8855    0.8849      7311
weighted avg     0.8939    0.8939    0.8931      7311

```
  [OK ] Accuracy ≥ 80 % (7311 commentaires)  — 89.39%
  [OK ] F1 par classe > 0,75  — nég=0.826 / neu=0.930 / pos=0.899

### 1.2 Cas limites

| Cas | Texte (extrait) | Prédiction | Confiance |
|---|---|---|---|
| Très court (1 lettre) | `a` | neutral | 0.97 |
| Très court (1 mot positif) | `great` | positive | 1.00 |
| Très court (1 mot négatif) | `awful` | negative | 0.99 |
| Vide | `` | neutral (texte vide) | 1.00 |
| Ponctuation seule | `!!!???` | neutral | 0.97 |
| Très long (~6 000 car.) | `this video is really amazing and helpful…` | positive | 1.00 |
| Emojis seuls positifs | `😍🔥👍` | positive | 1.00 |
| Emojis seuls négatifs | `😡👎` | negative | 1.00 |
| Texte + emoji | `thanks for this tutorial 😊` | positive | 1.00 |
| Majuscules / répétitions | `SOOOOO GOOOOD!!!!` | positive | 1.00 |
| URL + mention | `@john check https://youtu.be/xyz this is…` | negative | 1.00 |
| Mélange FR/EN | `C'est génial, really amazing work!` | positive | 1.00 |
| Français seul | `Cette vidéo est nulle, je déteste` | neutral | 0.96 |
| Arabe | `فيديو رائع جدا` | neutral | 0.97 |
| Chinois | `这个视频很好` | neutral | 0.97 |
| Négation | `this is not good at all` | positive | 1.00 |
| Sarcasme | `oh great, another useless video` | positive | 0.98 |

  [OK ] Aucun plantage sur 17 cas limites

  Remarque : le modèle est entraîné sur de l'anglais. Le français, l'arabe ou le chinois sont
  donc surtout classés « neutre » (mots inconnus). La négation (« not good ») et le sarcasme sont mal gérés :
  le modèle retenu utilise des unigrammes, il voit « good » sans « not ». Piste : bigrammes ou modèle Transformer.
  Ce sont des limites connues de l'approche TF-IDF.

### 1.3 Temps d'inférence (nettoyage + TF-IDF + prédiction)

| Taille du batch | Médiane (ms) | P95 (ms) |
|---|---|---|
| 1 | 0.92 | 1.58 |
| 10 | 1.18 | 2.11 |
| 50 | 2.80 | 4.31 |
| 100 | 3.71 | 4.91 |
| 500 | 15.44 | 17.46 |

  [OK ] Inférence batch de 50 < 100 ms  — 2.80 ms

## 2. Tests de l'API locale (uvicorn, src/api/main.py)

  [OK ] GET /health → 200 et modèle chargé  — {'status': 'ok', 'model_loaded': True, 'model_version': '20261104-0950'}
  [OK ] GET /docs → 200
  [OK ] CORS autorisé pour l'extension Chrome

  Temps de réponse (API locale) :

| Batch | Réseau + API, médiane (ms) | Temps modèle côté serveur (ms) | Statut |
|---|---|---|---|
| 1 | 4.0 | 0.66 | OK |
| 10 | 22.3 | 3.31 | OK |
| 50 | 18.7 | 2.31 | OK |
| 100 | 21.5 | 6.17 | OK |
| 500 | 11.3 | 7.14 | OK |

  [OK ] POST /predict_batch, batchs de 1 à 500
  [OK ] Erreur gérée : batch vide → 422  — code 422
  [OK ] Erreur gérée : que des textes vides → 422  — code 422
  [OK ] Erreur gérée : mauvais type → 422  — code 422
  [OK ] Erreur gérée : mauvais champ → 422  — code 422
  [OK ] Erreur gérée : corps vide → 422  — code 422
  [OK ] Erreur gérée : batch > 500 → 422  — code 422
  [OK ] Erreur gérée : élément non-texte → 422  — code 422
  [OK ] Erreur gérée : JSON malformé → 422  — code 422

## 3. Tests de l'API déployée (Docker sur GitHub Codespaces, app_api.py)

  [OK ] GET /health → 200 et modèle chargé  — {'status': 'ok', 'model_loaded': True, 'model_version': '1.0.0'}
  [OK ] GET /docs → 200
  [OK ] CORS autorisé pour l'extension Chrome

  Temps de réponse (API cloud) :

| Batch | Réseau + API, médiane (ms) | Temps modèle côté serveur (ms) | Statut |
|---|---|---|---|
| 1 | 1202.8 | 1.01 | OK |
| 10 | 1163.0 | 1.13 | OK |
| 50 | 1092.0 | 1.62 | OK |
| 100 | 1289.4 | 2.58 | OK |
| 500 | 1514.4 | 6.5 | OK |

  [OK ] POST /predict_batch, batchs de 1 à 500
  [OK ] Erreur gérée : batch vide → 422  — code 422
  [OK ] Erreur gérée : que des textes vides → 422  — code 422
  [OK ] Erreur gérée : mauvais type → 422  — code 422
  [OK ] Erreur gérée : mauvais champ → 422  — code 422
  [OK ] Erreur gérée : corps vide → 422  — code 422
  [OK ] Erreur gérée : batch > 500 → 422  — code 422
  [OK ] Erreur gérée : élément non-texte → 422  — code 422
  [OK ] Erreur gérée : JSON malformé → 422  — code 422

## 4. Intégration : cohérence entre API locale et API cloud

| Commentaire | Locale | Cloud |
|---|---|---|
| I love this video, great explanation! | positive (1.00) | positive (1.00) |
| This is the worst tutorial ever | negative (1.00) | negative (1.00) |
| What camera do you use? | neutral (0.97) | neutral (0.97) |
| 😍🔥 | positive (1.00) | positive (1.00) |
| meh | neutral (0.83) | neutral (0.83) |
| not bad at all | negative (1.00) | negative (1.00) |
| I hate it | negative (1.00) | negative (1.00) |

  [OK ] Prédictions identiques en local et en cloud
  [OK ] Statistiques identiques

## Synthèse

| Catégorie | Tests réussis |
|---|---|
| Modèle | 4/4 |
| API locale | 12/12 |
| API cloud | 12/12 |
| Intégration | 2/2 |
| **Total** | **30/30** |
