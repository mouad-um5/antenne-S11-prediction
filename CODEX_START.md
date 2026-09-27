# Contexte pour Codex — Projet Antenne Prediction

## But du projet
Construire une application web moderne permettant de prédire le coefficient S11 d'une antenne à partir d'un modèle XGBoost entraîné sur `backend/data/pik_clean_ml.csv`.

## État actuel
Le projet contient déjà :
- un script d'entraînement reproductible : `backend/src/train_model.py` ;
- une API FastAPI : `backend/app/main.py` ;
- une couche d'inférence : `backend/app/predictor.py` ;
- une interface React/Vite dans `frontend/` ;
- les données originales et le document de référence `backend/docs/BOT.xlsx`.

## Variables ML actuelles
Features :
1. `gap`
2. `surface_width`
3. `surface_length`
4. `epsilon_r`
5. `FREQUENCY`

Target : `s11` en dB.

## Plages du dataset
- epsilon_r : 1.0 — 5.0
- gap : 26.0 — 49.0
- surface_length : 40.0 — 110.0
- surface_width : 40.0 — 110.0
- frequency : 0.5 — 4.0 GHz

## API existante
- `GET /api/health`
- `GET /api/model-info`
- `POST /api/predict`
- `POST /api/predict-sweep`

## Priorités de développement
1. Ne pas casser l'API et le modèle existants.
2. Ajouter une validation des valeurs par rapport aux plages du dataset, avec warning d'extrapolation.
3. Améliorer l'interface utilisateur et le graphique S11.
4. Afficher automatiquement : S11 minimum, fréquence associée, zones sous -10 dB et bande passante estimée.
5. Ajouter une page expliquant les paramètres d'antenne à partir de `backend/docs/BOT.xlsx` sans mélanger ces informations avec les features du modèle actuel.
6. Préparer une architecture permettant d'ajouter plus tard d'autres types d'antennes et d'autres modèles/outputs CST.
7. Ajouter tests unitaires et tests API.
8. Ajouter Docker uniquement après stabilisation de l'application locale.

## Contraintes importantes
- Le modèle actuel a été entraîné uniquement sur les 5 features listées ci-dessus.
- Ne pas inventer de features supplémentaires pour l'inférence sans réentraîner un nouveau modèle.
- Ne pas présenter une prédiction hors plage comme fiable : afficher explicitement qu'il s'agit d'une extrapolation.
- Conserver le nom exact de la feature CSV `FREQUENCY` dans la DataFrame envoyée au modèle.
- Le fichier `BOT.xlsx` est une base documentaire sur les familles d'antennes et relations physiques, pas une source directe de features pour le modèle S11 actuel.

## Première mission suggérée à Codex
Analyse tout le projet existant, exécute les tests manuels nécessaires, puis améliore l'interface sans modifier le comportement mathématique du modèle. Ajoute ensuite la validation des plages d'entraînement, les avertissements d'extrapolation et le calcul de la bande où S11 <= -10 dB à partir du sweep prédit.
