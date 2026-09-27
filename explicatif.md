# Note explicative — Projet Antenne Prediction

## 1. Résumé du projet

**Antenne Prediction** est un prototype web de prédiction du coefficient de réflexion **S11** d'une antenne. L'utilisateur fournit quatre paramètres géométriques ou matériaux ainsi qu'une fréquence, puis une régression XGBoost estime la valeur de S11 en dB.

L'application permet actuellement de :

- choisir obligatoirement une famille et une variante d'antenne parmi le catalogue issu de `BOT.xlsx` ;
- prédire S11 à une fréquence précise ;
- générer une courbe S11 sur une plage de fréquences ;
- afficher le minimum prédit de la courbe et la fréquence correspondante ;
- calculer les bandes passantes sous un seuil S11 configurable ;
- signaler les valeurs hors plage et les géométries éloignées du jeu d'entraînement ;
- afficher l'erreur P95 issue de la validation groupée ;
- vérifier l'intégrité et la traçabilité des artefacts du modèle.

Il faut présenter ce système comme un **modèle de substitution** rapide pour explorer les données disponibles, et non comme un solveur électromagnétique, un jumeau numérique validé ou un remplacement automatique de CST et des mesures expérimentales.

## 2. Contexte scientifique

S11 décrit la réflexion au port d'entrée de l'antenne. Une valeur plus négative traduit généralement une meilleure adaptation à cette fréquence. Le seuil de -10 dB, affiché dans l'interface, est un repère couramment utilisé, mais il doit être confirmé pour le cas d'étude concerné.

S11 seul ne permet pas de conclure sur toutes les performances de l'antenne. Il ne remplace notamment pas l'étude de l'impédance, du gain, du rendement, de la directivité, du diagramme de rayonnement, de la polarisation ou du comportement réel après fabrication.

Le modèle actuel utilise exactement les variables suivantes :

| Variable | Signification utilisée par l'application | Domaine observé pendant l'entraînement |
|---|---|---:|
| `gap` | Espacement géométrique | 26 à 49, unité à confirmer |
| `surface_width` | Largeur du substrat | 40 à 110, unité à confirmer |
| `surface_length` | Longueur du substrat | 40 à 110, unité à confirmer |
| `epsilon_r` | Permittivité relative | 1 à 5, sans unité |
| `FREQUENCY` | Fréquence | 0,5 à 4 GHz |
| `s11` | Grandeur prédite | S11 en dB |

Les unités du `gap`, de la largeur et de la longueur ne sont pas définies dans le code ni dans l'interface. Elles doivent être confirmées par la responsable scientifique avant toute diffusion des résultats.

## 3. Fonctionnement pour l'utilisateur

### Parcours guidé en quatre étapes

L'interface se présente désormais comme un assistant de simulation en quatre étapes numérotées :

1. **Antenne** : choix obligatoire de la grande famille et de la variante ;
2. **Paramètres** : saisie de la géométrie et de la fréquence ponctuelle ;
3. **Résultat** : affichage de la valeur S11 prédite et des avertissements ;
4. **Courbe S11** : configuration et génération du balayage fréquentiel.

Les étapes sont reliées visuellement par des segments qui s'arrêtent au bord des cercles numérotés. Les étapes non encore disponibles sont désactivées. L'utilisateur peut revenir vers une étape accessible sans perdre sa sélection ni les valeurs déjà saisies. Une modification de l'antenne ou des paramètres invalide les résultats concernés afin d'éviter d'afficher une ancienne prédiction comme si elle correspondait aux nouvelles valeurs.

### Choix de l'antenne

Avant de saisir les paramètres, l'utilisateur choisit une grande famille puis une variante. Le catalogue comporte actuellement 16 familles et 39 variantes reprises de `backend/docs/BOT.xlsx`.

Cette sélection est envoyée à l'API et vérifiée contre le catalogue. À ce stade, le modèle n'est associé à aucune famille scientifiquement confirmée : en développement, l'API retourne donc un avertissement avec chaque résultat ; en production, elle bloque la prédiction par défaut.

Le catalogue n'est plus copié manuellement dans React. Le backend lit directement la feuille `grand type antenne` de `backend/docs/BOT.xlsx`, construit les 16 familles et 39 variantes, puis les expose avec `GET /api/antennas`.

### Prédiction ponctuelle

L'utilisateur saisit une géométrie et une fréquence. Le navigateur envoie ces cinq valeurs à l'API, qui les transmet au modèle et retourne :

- `s11_db` : estimation de S11 ;
- `is_extrapolation` : indique si la demande sort du domaine détecté ;
- `known_training_geometry` et `nearest_geometry_distance` : proximité avec les géométries apprises ;
- `estimated_absolute_error_p95_db` : erreur absolue P95 observée en validation interne ;
- `model_id`, `scientific_status` et `model_scope_verified` : traçabilité et statut scientifique ;
- `warnings` : avertissements de domaine et de périmètre scientifique.

### Balayage fréquentiel

L'utilisateur saisit une géométrie, une fréquence de début, une fréquence de fin, un nombre de points et un seuil S11. L'API construit une grille de fréquences, prédit S11 pour chaque point, retourne la courbe et son minimum, puis interpole les franchissements du seuil afin de calculer chaque bande et la bande passante totale.

Le « minimum prédit » est donc un minimum **échantillonné**, pas nécessairement le minimum continu exact. Sa précision dépend du nombre de points demandé.

### Ce que l'application ne fait pas encore

- elle ne fournit pas encore un intervalle de confiance calibré ; la valeur P95 affichée résume les erreurs de validation, mais n'est pas un intervalle individuel garanti ;
- elle ne compare pas une courbe prédite à une courbe CST ou mesurée ;
- elle ne mémorise pas les expériences ni les résultats ;
- elle n'adapte pas encore les paramètres de saisie à la famille choisie ;
- elle ne sélectionne pas encore un modèle ML différent selon l'antenne ;
- elle n'explique pas encore toutes les relations physiques présentes dans `backend/docs/BOT.xlsx`.

## 4. Architecture du projet

```text
Navigateur
   |
   | Interface React + graphique Chart.js
   | requêtes JSON vers /api
   v
API FastAPI
   |
   +--> lecture du catalogue docs/BOT.xlsx
   +--> validation, contrôle du périmètre et analyse des bandes
   v
Couche d'inférence Python
   |
   +--> modèle sérialisé models/s11_predictor.pkl
   +--> métadonnées models/model_info.json

Entraînement hors ligne
data/pik_clean_ml.csv
   |
   v
src/train_model.py --> audit + agrégation + GroupKFold + XGBoost
                    --> modèle .pkl + traçabilité complète .json

Livraison
Docker Compose --> Nginx/React --> FastAPI --> modèle vérifié
CI GitHub Actions --> tests backend + lint/build frontend
```

Les deux parties déployables sont séparées :

| Couche | Technologie | Rôle principal |
|---|---|---|
| Frontend | React 19, Vite, Chart.js | Formulaires, avertissements, résultat ponctuel et courbe S11 |
| Backend | FastAPI, Pydantic | API HTTP, validation des données et CORS |
| Inférence | pandas, NumPy, joblib | Construction des entrées et chargement du modèle en mémoire |
| Modèle | scikit-learn, XGBoost | Régression de S11 à partir des cinq variables |
| Données | CSV | Source d'entraînement |
| Tests | pytest/FastAPI TestClient | Tests élémentaires de l'API |
| Production | Docker, Docker Compose, Nginx | Build reproductible, proxy et healthchecks |
| CI | GitHub Actions | Tests, lint et build à chaque changement |

### Arborescence utile

```text
antenne_prediction_starter/
|-- backend/
|   |-- app/
|   |   |-- main.py              # routes, sécurité et schémas d'entrée/sortie
|   |   |-- predictor.py         # chargement, intégrité et domaine du modèle
|   |   |-- catalog.py           # extraction du catalogue Excel
|   |   |-- curve_analysis.py    # calcul des bandes sous le seuil
|   |   `-- settings.py          # configuration par environnement
|   |-- data/
|   |   `-- pik_clean_ml.csv     # données d'entraînement
|   |-- docs/
|   |   |-- BOT.xlsx             # documentation scientifique source
|   |   `-- predict_s11_original.py
|   |-- models/
|   |   |-- s11_predictor.pkl    # pipeline entraîné
|   |   `-- model_info.json      # variables, métriques et plages
|   |-- src/train_model.py       # entraînement hors ligne
|   |-- tests/                   # 11 tests API et calculs scientifiques
|   `-- Dockerfile
|-- frontend/
|   |-- src/App.jsx              # écran et orchestration principale
|   |-- src/api/client.js        # appels vers l'API
|   |-- src/components/          # champs numériques et graphique
|   |-- src/data/antennaCatalog.js # adresse de l'API catalogue
|   |-- nginx.conf
|   `-- Dockerfile
|-- .github/workflows/ci.yml
|-- docker-compose.yml
|-- README.md                    # installation et exécution
`-- explicatif.md                # présente note
```

### API actuelle

| Méthode et route | Fonction |
|---|---|
| `GET /` | Informations générales sur le service |
| `GET /api/health` | Vérifie que le processus API répond |
| `GET /api/ready` | Vérifie que le modèle peut être chargé |
| `GET /api/antennas` | Expose le catalogue lu depuis `BOT.xlsx` |
| `GET /api/model-info` | Retourne les variables, métriques et plages d'entraînement |
| `POST /api/predict` | Prédiction ponctuelle |
| `POST /api/predict-sweep` | Prédiction d'une courbe fréquentielle |

En développement, Vite transmet `/api` à FastAPI sur `127.0.0.1:8000`. En production, Nginx sert React et transmet `/api` au backend dans le réseau Docker. `CORS_ORIGINS`, `ALLOWED_HOSTS`, les docs API et le contrôle du périmètre sont configurables par environnement.

Le choix de la famille et de la variante est envoyé aux endpoints de prédiction. Il est validé contre Excel, puis comparé au périmètre enregistré dans `model_info.json`. Il reste absent des cinq variables mathématiques XGBoost : il sert de garde-fou et de contexte, pas de feature implicite.

## 5. Modèle de machine learning actuel

Le script `backend/src/train_model.py` :

1. lit le CSV ;
2. contrôle les colonnes, valeurs manquantes et nombres non finis ;
3. audite les doublons et les cibles contradictoires ;
4. regroupe les entrées identiques et moyenne leur cible, ce qui produit 31 031 entrées effectives ;
5. réserve des géométries complètes avec une validation `GroupKFold` en cinq plis ;
6. mesure les erreurs globales, par courbe, sur le minimum S11 et sur la fréquence du minimum ;
7. entraîne XGBoost sur les données nettoyées, sans scaler inutile ;
8. sauvegarde atomiquement le modèle, les empreintes SHA-256, les versions, les hyperparamètres et le statut scientifique.

Les métriques enregistrées sont :

| Métrique | Valeur enregistrée |
|---|---:|
| MAE groupée | 0,03807 dB |
| RMSE groupée | 0,10076 dB |
| R² groupé | 0,99896 |
| Erreur absolue P95 | 0,21305 dB |
| Erreur moyenne sur la fréquence du minimum | 0,00655 GHz |
| Erreur maximale sur la fréquence du minimum | 0,01750 GHz |

Ces valeurs sont plus défendables que l'ancien découpage aléatoire, car chaque pli teste des géométries absentes de son entraînement. Elles restent néanmoins des **métriques internes** calculées sur le même ensemble de simulations. Elles ne démontrent pas encore la concordance avec une campagne CST externe ou une antenne mesurée.

Le `StandardScaler` a été supprimé, car les arbres XGBoost n'en ont pas besoin.

## 6. Analyse du jeu de données

L'inspection du CSV donne les constats suivants :

- 40 040 lignes et 12 colonnes ;
- 1 001 fréquences distinctes ;
- seulement **31 combinaisons distinctes** de `gap`, largeur, longueur et `epsilon_r` ;
- 6 006 lignes complètement dupliquées ;
- 9 009 répétitions de combinaisons des cinq variables d'entrée ;
- 2 002 combinaisons d'entrée possèdent plusieurs cibles S11 différentes, avec jusqu'à trois valeurs pour une même entrée ;
- ces conflits se concentrent sur deux configurations géométriques ;
- `surface_width` est égal à `surface_length` dans toutes les lignes ;
- aucune valeur manquante n'a été détectée ;
- certaines colonnes non utilisées sont constantes, répétées ou ont une signification à clarifier.

Ces résultats entraînent plusieurs conséquences importantes :

1. **Les lignes ne sont pas des observations indépendantes.** Une grande partie correspond aux points fréquentiels successifs d'une même courbe.
2. **L'ancien découpage aléatoire par ligne mélangeait les mêmes géométries entre entraînement et test.** Il a été remplacé par une validation groupée qui réserve des géométries complètes.
3. **Les doublons ne sont plus utilisés comme des observations supplémentaires.** Les 9 009 répétitions d'entrée sont regroupées avant l'entraînement ; les cibles en conflit sont moyennées et documentées.
4. **La largeur et la longueur ne sont pas identifiables séparément** à partir de ces données, puisqu'elles varient toujours ensemble.
5. **Un contrôle par minimum/maximum ne suffit pas.** Une largeur de 40 et une longueur de 110 sont chacune dans leur plage, mais leur combinaison n'apparaît jamais dans les données. L'application peut donc la traiter comme une interpolation alors qu'elle est hors distribution.

L'interface affiche désormais « 31 031 entrées uniques · 31 géométries », formulation plus fidèle que l'ancien nombre de 40 040 observations.

## 7. Points forts

### Sur le plan logiciel

- séparation nette entre frontend, API, inférence et entraînement ;
- chemins de données et de modèle relatifs au projet, sans dépendance à un poste particulier ;
- modèle chargé une seule fois en mémoire grâce à un cache ;
- validation des nombres finis, des valeurs positives et du nombre de points ;
- avertissements d'extrapolation renvoyés par l'API, et pas seulement calculés dans l'interface ;
- endpoints de santé et de disponibilité adaptés à un futur déploiement ;
- CORS configurable par variable d'environnement ;
- prédiction du sweep vectorisée, plus efficace qu'une requête par fréquence ;
- interface lisible avec courbe, minimum prédit et repère -10 dB ;
- parcours guidé en quatre étapes avec sélection obligatoire de l'antenne ;
- présence d'un script d'entraînement reproductible et de premiers tests API.

### Sur le plan du prototype scientifique

- variables et cible clairement séparées ;
- plage fréquentielle dense dans les données ;
- graine aléatoire fixée pour le découpage et XGBoost ;
- métriques et domaines marginaux sauvegardés avec le modèle ;
- l'application avertit déjà l'utilisateur lorsqu'une extrapolation évidente est demandée.

## 8. Points faibles et risques

### Priorité critique — validité scientifique restante

- **Peu de géométries indépendantes :** 31 configurations sont insuffisantes pour revendiquer une généralisation large dans un espace à quatre paramètres géométriques/matériaux.
- **Cibles contradictoires :** elles sont maintenant auditées et agrégées, mais leur origine physique doit encore être expliquée.
- **Largeur et longueur confondues dans les données :** le modèle ne peut pas apprendre correctement leur effet indépendant.
- **Aucune validation externe :** le dépôt ne contient pas de comparaison sur des simulations totalement réservées ni sur des mesures physiques.
- **Incertitude non calibrée :** l'interface affiche l'erreur P95 et la distance à la géométrie la plus proche, mais pas encore un intervalle prédictif individuel calibré.
- **Famille du dataset inconnue :** la sélection est validée contre le catalogue, mais aucune des 39 variantes ne peut encore être déclarée scientifiquement supportée.
- **Aucune contrainte physique :** XGBoost prédit chaque point indépendamment et ne garantit ni continuité fréquentielle ni comportement physiquement admissible.

### Priorité importante — interprétation des résultats

- les unités géométriques et le référentiel exact des paramètres ne sont pas documentés ;
- le type et la topologie de l'antenne, les conditions aux limites, le port, le maillage et la version du simulateur ne sont pas associés au modèle ;
- le minimum dépend de la résolution du sweep ;
- la courbe affichée utilise un léger lissage graphique, qui ne doit pas être confondu avec une donnée simulée ;
- les bandes sous le seuil sont interpolées entre les points prédits et restent donc dépendantes de la résolution du sweep et de la qualité du modèle ;
- S11 ne suffit pas à qualifier globalement une antenne.

### Priorité importante — robustesse logicielle

- onze tests backend couvrent l'API, le catalogue, le domaine et le calcul des bandes, mais aucun test frontend ou test navigateur complet n'est encore présent ;
- les dépendances de production Python sont verrouillées, mais leur mise à jour de sécurité doit être organisée ;
- les métadonnées contiennent date, versions, hyperparamètres et empreintes du dataset/modèle, mais pas encore l'identifiant du commit Git ;
- le modèle courant est inclus explicitement malgré la règle générale d'exclusion des `.pkl` ; un registre d'artefacts restera préférable lorsque plusieurs modèles existeront ;
- un fichier joblib/pickle ne doit être chargé que depuis une source de confiance ;
- la CI, Docker, Nginx, les healthchecks et les journaux de requêtes sont présents ; il manque encore un suivi de dérive, un registre de modèles, des sauvegardes et une supervision externe ;
- si l'API devient publique, il faudra ajouter les protections adaptées : HTTPS, limitation de débit, politique d'accès et supervision.

## 9. Améliorations recommandées

### P0 — rendre l'évaluation scientifiquement défendable

1. Identifier la famille et la variante exactes correspondant au CSV actuel.
2. Définir précisément chaque variable, son unité, son mode de mesure et la provenance des données.
3. Expliquer les 2 002 entrées possédant plusieurs cibles S11 : répétitions valides, simulations différentes ou erreurs de fusion.
4. Donner un identifiant stable à chaque configuration, simulation et courbe.
5. Conserver un jeu de validation externe bloqué, idéalement avec de nouvelles simulations CST puis des mesures réelles.
6. Ajouter l'erreur de bande passante aux métriques de validation lorsque les courbes de référence sont qualifiées.
7. Comparer XGBoost à des références simples afin de prouver son apport.

### P1 — améliorer le produit scientifique

1. Afficher la résolution fréquentielle du sweep et préciser que le minimum est discret.
2. Afficher les configurations d'entraînement les plus proches, en complément de la distance déjà calculée.
3. Fournir une estimation d'incertitude calibrée, par exemple via bootstrap, quantiles ou conformal prediction, après définition d'un protocole valide.
4. Ajouter l'import d'une courbe CST/mesurée pour superposition et calcul automatique des erreurs.
5. Enrichir le catalogue avec les géométries, matériaux, paramètres de sweep et outputs présents dans `BOT.xlsx`, après validation scientifique de son contenu.
6. Si largeur et longueur doivent varier indépendamment, produire de nouvelles simulations couvrant réellement ces deux axes avant de conserver deux champs indépendants dans l'interface.
7. Définir pour chaque famille son propre schéma d'entrée et son modèle validé ; jusque-là, conserver le blocage de production.

### P2 — fiabiliser l'exploitation

1. Ajouter des tests frontend et un test navigateur du parcours complet.
2. Ajouter un test de non-régression sur un petit jeu scientifique de référence indépendant.
3. Ajouter le commit du code aux métadonnées générées dans un véritable dépôt Git.
4. Choisir un registre d'artefacts lorsque plusieurs modèles seront disponibles.
5. Déployer derrière HTTPS et mettre en place supervision, alertes, sauvegardes et procédure de retour arrière.
6. Ajouter limitation de débit et authentification si l'API doit être exposée au public.

## 10. Répartition des responsabilités

| Responsable scientifique / doctorante | Développeur | Décisions communes |
|---|---|---|
| Définir les unités et le sens physique des variables | Implémenter l'interface et l'API | Définir les critères d'acceptation |
| Garantir la provenance et la qualité des données | Assurer tests, déploiement et traçabilité | Choisir ce qui peut être présenté comme fiable |
| Valider le protocole train/validation/test | Empêcher ou signaler les usages hors domaine | Versionner ensemble données, modèle et application |
| Fournir des cas CST/mesurés jamais vus | Afficher clairement limites et incertitudes | Décider du seuil S11 et des métriques métier |
| Interpréter physiquement les erreurs et résultats | Reproduire automatiquement les calculs | Autoriser ou interdire l'extrapolation |

## 11. Formulation conseillée pour présenter l'outil

> Ce prototype estime rapidement la réponse S11, dans le domaine couvert par les données disponibles, à partir d'un modèle XGBoost évalué par géométries complètes. Ses résultats restent issus d'une validation interne et doivent être confirmés par une campagne CST externe puis, lorsque nécessaire, par mesure.

À éviter pour le moment : « le modèle remplace CST », « précision de 99,9 % », « valide pour toute antenne » ou « résultat expérimental ». Le R² de 0,99896 n'est pas une précision en pourcentage et ne remplace pas une validation externe.

## 12. Conclusion

Le projet dispose maintenant d'une base technique de production : responsabilités séparées, données auditées, validation groupée, artefacts traçables, contrôles de domaine, tests, CI et conteneurs avec healthchecks. L'application calcule aussi les bandes sous un seuil S11 configurable.

La limite restante est scientifique et ne peut pas être résolue par du code seul : il faut identifier l'antenne du CSV, confirmer les unités et confronter le modèle à des simulations ou mesures réellement indépendantes. La production bloque volontairement un modèle dont ce périmètre n'est pas attesté.
