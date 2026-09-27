# Antenne Prediction

Application web de prédiction S11 composée d'un backend FastAPI, d'un modèle
XGBoost et d'une interface React/Vite.

## Statut scientifique

Le modèle a été réentraîné après agrégation des entrées répétées et évalué avec
une validation croisée en 5 plis qui réserve des **géométries complètes** au
test. Les métriques internes actuelles sont :

- MAE : 0,0381 dB ;
- RMSE : 0,1008 dB ;
- R² : 0,99896 ;
- erreur absolue P95 : 0,2130 dB ;
- erreur moyenne sur la fréquence du minimum : 0,0065 GHz.

Ces résultats portent sur les simulations du CSV. Ils ne constituent pas encore
une validation expérimentale. La famille d'antenne du dataset n'étant pas
identifiée, la production bloque les prédictions par défaut.

## Fonctionnalités

- catalogue dynamique de 16 familles et 39 variantes lu depuis BOT.xlsx ;
- sélection obligatoire de l'antenne ;
- prédiction S11 ponctuelle ;
- sweep fréquentiel vectorisé ;
- minimum S11 et fréquence associée ;
- calcul interpolé des bandes sous un seuil configurable ;
- avertissements min/max et détection de géométries éloignées ;
- avertissement lorsque largeur et longueur sortent de la relation apprise ;
- traçabilité du modèle, des données et des dépendances ;
- healthcheck, readiness, request ID et en-têtes de sécurité ;
- frontend et backend conteneurisés ;
- CI pour tests, lint et build.

## Structure

~~~text
antenne_prediction_starter/
|-- backend/
|   |-- app/                 # API, catalogue, inférence et analyse des courbes
|   |-- data/                # dataset d'entraînement
|   |-- docs/BOT.xlsx        # catalogue scientifique source
|   |-- models/              # modèle et métadonnées vérifiables
|   |-- src/train_model.py   # audit, validation groupée et entraînement
|   |-- tests/               # tests API et calculs de bande
|   |-- Dockerfile
|   |-- requirements.txt     # contraintes de développement
|   `-- requirements.lock   # versions exactes de production
|-- frontend/
|   |-- src/
|   |-- nginx.conf           # SPA et proxy /api
|   `-- Dockerfile
|-- .github/workflows/ci.yml
|-- docker-compose.yml
|-- explicatif.md
`-- README.md
~~~

## Exécution locale

### Backend

~~~powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
uvicorn app.main:app --reload
~~~

API : http://127.0.0.1:8000  
Documentation locale : http://127.0.0.1:8000/docs

### Frontend

Node.js 20.19 ou plus récent est requis.

~~~powershell
cd frontend
npm ci
npm run dev
~~~

Frontend : http://localhost:5173

## Tests et contrôles

~~~powershell
cd backend
..\venv\Scripts\python.exe -B -m pytest

cd ..\frontend
npm run lint
npm run build
~~~

## Réentraînement et association scientifique

Un entraînement sans famille reste utilisable en développement, mais conserve
le statut internal_grouped_validation_only et un périmètre non vérifié :

~~~powershell
cd backend
python src/train_model.py
~~~

Après confirmation scientifique de l'antenne réellement représentée par le CSV,
réentraîner avec l'identifiant retourné par GET /api/antennas :

~~~powershell
python src/train_model.py --antenna-family-id "ID_FAMILLE" --antenna-variant "NOM_VARIANTE" --scope-verified
~~~

L'option --scope-verified est une attestation : elle ne doit être utilisée
qu'après vérification de la provenance du dataset et de ses conditions CST.

## Déploiement Docker

1. Copier la configuration :

~~~powershell
Copy-Item .env.production.example .env.production
~~~

2. Configurer le domaine public et conserver
   REQUIRE_VERIFIED_MODEL_SCOPE=true.

3. Vérifier que le modèle a été entraîné avec le bon périmètre.

4. Construire et lancer :

~~~powershell
docker compose --env-file .env.production up --build -d
docker compose ps
~~~

Le site est exposé sur http://localhost:8080 par défaut. En production,
terminer TLS/HTTPS devant Nginx et sauvegarder/versionner les artefacts du modèle.

Pour une démonstration technique uniquement, il est possible de définir
REQUIRE_VERIFIED_MODEL_SCOPE=false. L'API retournera alors un avertissement
scientifique avec chaque prédiction.

## Variables de production

| Variable | Rôle | Valeur sûre recommandée |
|---|---|---|
| APP_PORT | Port public du frontend | 8080 |
| CORS_ORIGINS | Origines navigateur autorisées | domaine public exact |
| ENABLE_API_DOCS | Expose Swagger/ReDoc | false |
| REQUIRE_VERIFIED_MODEL_SCOPE | Bloque un modèle non associé | true |
| ALLOWED_HOSTS | Hôtes acceptés par FastAPI | domaines/proxy exacts |
| MODEL_PATH | Modèle alternatif | chemin contrôlé |
| MODEL_INFO_PATH | Métadonnées alternatives | chemin contrôlé |

## Endpoints

| Route | Fonction |
|---|---|
| GET /api/health | Processus API disponible |
| GET /api/ready | Modèle, empreinte, catalogue et périmètre prêts |
| GET /api/antennas | Catalogue extrait de BOT.xlsx |
| GET /api/model-info | Métriques, audit et traçabilité |
| POST /api/predict | Prédiction ponctuelle |
| POST /api/predict-sweep | Courbe, minimum et bandes sous le seuil |

## Condition indispensable pour des résultats réels

Le fichier backend/data/external_validation_template.csv définit le format
attendu. Une validation externe sans réentraînement se lance avec :

~~~powershell
cd backend
python src/validate_external.py data/validation_cst.csv --validation-kind cst-unseen --max-mae-db 0.25 --max-resonance-error-ghz 0.03
~~~

Le protocole complet est décrit dans backend/docs/SCIENTIFIC_VALIDATION.md.

Pour qualifier les résultats de réels ou validés, il reste nécessaire de fournir :

1. la famille et la variante exactes du dataset actuel ;
2. les unités des dimensions ;
3. les réglages et la provenance des simulations CST ;
4. un jeu de géométries jamais utilisé pendant le développement ;
5. idéalement des mesures sur antennes fabriquées.

L'application fournit désormais le protocole, les garde-fous et la traçabilité
pour intégrer ces données sans confondre prédiction ML et mesure physique.
