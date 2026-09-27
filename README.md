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
- choix entre formulaire et paragraphe avec extraction Gemini, Groq, Cloudflare ou Mistral et relecture ;
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

## Saisie par paragraphe avec Gemini, Groq, Cloudflare ou Mistral

Le bouton **Décrire avec un paragraphe** envoie le texte au fournisseur sélectionné côté serveur,
puis préremplit les champs pour relecture. Les valeurs absentes ou invalides restent
vides, y compris les réglages du balayage. L'utilisateur complète les champs et
déclenche lui-même la prédiction S11. Le modèle choisi extrait les paramètres ; XGBoost
continue à effectuer les prédictions.

Un sélecteur propose six modèles :
- **Gemini 3.6 Flash** (`gemini-3.6-flash`), le modèle initial ;
- **Gemini 3.8 Flash** (`gemini-3.8-flash`), une version récente ;
- **Gemini 3.5 Flash-Lite** (`gemini-3.5-flash-lite`), une version légère ;
- **Groq · GPT-OSS 20B** (`openai/gpt-oss-20b`), avec sortie JSON structurée ;
- **Cloudflare · Llama 3.1 8B Fast** (`@cf/meta/llama-3.1-8b-instruct-fast`) ;
- **Mistral · Mistral Small** (`mistral-small-latest`), avec sortie JSON structurée.

Les trois modèles Gemini utilisent `GEMINI_API_KEY`. Groq utilise une clé séparée,
`GROQ_API_KEY`, et son endpoint officiel ; aucune clé Gemini n'est envoyée à Groq.

Les fournisseurs proposent des offres gratuites, sous réserve des quotas, du plan
activé et des accès du compte. Vérifier le plan et les limites dans leur console. `GEMINI_MODEL` (nom historique) définit le choix initial
et doit être l'un de ces six identifiants. Le choix de l'utilisateur s'applique à chaque extraction.
L'API refuse tout autre identifiant et renvoie le modèle effectivement utilisé.
Le bouton **Retester le paragraphe avec un autre modèle** conserve le texte pour
un nouvel essai. Les valeurs du nouvel essai remplacent le préremplissage précédent ;
aucun basculement automatique vers un autre modèle n'est effectué.

1. Créer une clé dans [Google AI Studio](https://aistudio.google.com/apikey).
2. Choisir un projet bénéficiant de l'offre gratuite. La gratuité et les quotas
   dépendent du projet Google ; l'application ne contrôle pas sa facturation.
3. Dans le dossier backend, copier `.env.example` vers `.env` et renseigner
   `GEMINI_API_KEY`. Le modèle par défaut est `gemini-3.6-flash`.
4. Démarrer avec `uvicorn app.main:app --reload --port 8001`. Le backend charge
   automatiquement son fichier `.env`, sans remplacer les variables déjà définies
   dans le processus. Redémarrer le serveur après toute modification de `.env`.

Pour activer Groq :
1. Créer une clé dans [Groq Console](https://console.groq.com/keys), avec l'offre Free.
2. Ajouter `GROQ_API_KEY=votre_cle` dans `backend/.env`.
3. Redémarrer le backend puis sélectionner **Groq · GPT-OSS 20B** dans l'interface.

La clé Groq reste uniquement côté serveur. Aucun abonnement OpenAI ni clé xAI
n'est nécessaire : le modèle est hébergé par Groq. Sans clé Groq, cette option
affiche un message de configuration ; Gemini reste utilisable avec sa propre clé.
Les [quotas gratuits Groq](https://console.groq.com/docs/rate-limits) et les
[sorties structurées](https://console.groq.com/docs/structured-outputs) sont
documentés par le fournisseur. L'application n'active aucun abonnement payant.

Pour activer Cloudflare :
1. Dans [Cloudflare Workers AI](https://developers.cloudflare.com/workers-ai/get-started/rest-api/),
   récupérer l'Account ID et créer un jeton API autorisé à utiliser Workers AI sur ce compte.
2. Ajouter dans `backend/.env` :
   `CLOUDFLARE_ACCOUNT_ID=votre_identifiant_de_compte` et
   `CLOUDFLARE_API_TOKEN=votre_jeton` (deux lignes séparées).
3. Redémarrer le backend puis sélectionner **Cloudflare · Llama 3.1 8B Fast**.

L'Account ID est un identifiant hexadécimal de 32 caractères, pas un Zone ID.
Le backend appelle l'API REST Workers AI directement. Ce modèle Fast reçoit
une consigne JSON et le schéma dans le prompt ; la réponse est validée côté serveur.
La sortie JSON n'est pas garantie par le modèle : une réponse invalide produit
une erreur et ne préremplit pas le formulaire.
Les [quotas Workers AI](https://developers.cloudflare.com/workers-ai/platform/pricing/)
sont comptés en neurones, pas en nombre fixe de tokens.

Pour activer Mistral :
1. Créer une clé dans [Mistral Studio](https://console.mistral.ai/) avec l'offre gratuite
   activée et vérifier les [limites du compte](https://docs.mistral.ai/admin/billing-usage/usage-limits).
2. Ajouter `MISTRAL_API_KEY=votre_cle` dans `backend/.env`.
3. Redémarrer le backend puis sélectionner **Mistral · Mistral Small**.

Mistral utilise `https://api.mistral.ai/v1/chat/completions` et une sortie JSON
contrainte par schéma. L'alias `mistral-small-latest` suit les mises à jour du fournisseur.
Les identifiants Cloudflare et Mistral restent côté serveur et ne sont envoyés
qu'au fournisseur sélectionné. Aucune clé d'un autre fournisseur n'est nécessaire.
Un compte payant peut être facturé selon son plan ; l'application ne gère pas la facturation.

En cas d'erreur :
- **Cloudflare, authentification HTTP 401** (renvoyée par l'application en 503) :
  depuis Workers AI > Use REST API, créer un jeton Workers AI et copier l'Account ID
  du même compte. Un jeton personnalisé doit avoir les permissions Workers AI Read
  et Edit. Corriger `backend/.env` puis redémarrer le serveur.
- **Mistral, HTTP 429** : vérifier les limites de requêtes/tokens et l'utilisation
  du compte dans Mistral Studio. Respecter le délai affiché s'il est fourni ;
  sinon, aucun délai de reprise précis ne peut être déduit de cette réponse.
  L'application ne réessaie pas automatiquement.
- Ne renseigner les secrets que dans `.env` ou les variables du processus ;
  les fichiers `.env.example` doivent rester vides de clés.

Pour Docker, renseigner les mêmes variables dans `.env.production` puis lancer
la commande Docker Compose documentée ci-dessous. Ne jamais placer la clé dans
une variable `VITE_*`, dans le frontend ou dans Git.

Sans clé, le formulaire reste utilisable et l'extraction affiche un message de
configuration. Les quotas dépassés, délais et réponses invalides sont signalés
sans lancer de prédiction ni réessayer automatiquement.

Exemple de texte :

> Je souhaite étudier un dipôle demi-onde de la famille des antennes filaires,
> avec un gap de 39,47, une largeur de substrat de 75, une longueur de 75,
> une permittivité relative de 5 et une fréquence de 2,4 GHz. Je veux une courbe
> de 0,5 à 4 GHz sur 301 points avec un seuil de -10 dB.

Les fréquences explicitement unitaires sont demandées en GHz au modèle. Les
dimensions sont conservées sans conversion car leurs unités ne sont pas encore
documentées dans le dataset ; un message invite à les vérifier.

Documentation : [modèle](https://ai.google.dev/gemini-api/docs/models/gemini-3.6-flash),
[tarifs et offre gratuite](https://ai.google.dev/gemini-api/docs/pricing#gemini-3.6-flash),
[sorties structurées](https://ai.google.dev/gemini-api/docs/generate-content/structured-output).

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
| GET /api/extraction-models | Six modèles proposés, fournisseurs et choix par défaut |
| POST /api/extract-inputs | Extraction des champs depuis un paragraphe (Gemini, Groq, Cloudflare ou Mistral) |
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
