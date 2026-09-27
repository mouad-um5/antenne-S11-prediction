# Validation scientifique du modèle S11

## Ce qui est déjà validé

Le pipeline vérifie le schéma des données, agrège les entrées répétées et
effectue une validation croisée en cinq plis par géométrie complète. Le modèle,
le dataset, les hyperparamètres et les versions logicielles sont traçables dans
models/model_info.json.

Cette validation est interne : toutes les courbes viennent encore du même CSV.

## Informations obligatoires à fournir

- famille et variante exactes de l'antenne ;
- unité de gap, surface_width et surface_length ;
- signification géométrique exacte de chaque variable ;
- origine de chaque courbe ;
- version de CST ou de l'instrument ;
- paramètres de solveur, port, maillage et conditions aux limites ;
- explication des répétitions et des cibles S11 différentes pour une même entrée.

## Protocole CST externe

1. Créer des géométries absentes de pik_clean_ml.csv.
2. Ne pas ajuster le modèle après avoir vu leurs résultats.
3. Exporter toutes les fréquences et S11 dans une copie de
   data/external_validation_template.csv.
4. Exécuter depuis backend :

~~~powershell
python src/validate_external.py data/validation_cst.csv --validation-kind cst-unseen --max-mae-db 0.25 --max-resonance-error-ghz 0.03 --max-bandwidth-error-ghz 0.05
~~~

Le script refuse une validation CST dite « inédite » si une géométrie existe
déjà dans l'entraînement. Il produit un rapport JSON sans réentraîner le modèle.

## Protocole de mesure

Les mesures peuvent utiliser une géométrie déjà simulée si elles proviennent
réellement d'une antenne fabriquée et d'un instrument documenté :

~~~powershell
python src/validate_external.py data/validation_mesures.csv --validation-kind measurement --max-mae-db 0.50 --max-resonance-error-ghz 0.05
~~~

Les seuils ci-dessus sont des exemples techniques, pas des critères
scientifiques imposés. La doctorante doit fixer les seuils acceptables avant de
consulter le résultat final.

## Passage en production scientifique

Après validation de la provenance du CSV et confirmation de son antenne :

~~~powershell
python src/train_model.py --antenna-family-id "ID_FAMILLE" --antenna-variant "NOM_VARIANTE" --scope-verified
~~~

Ensuite, conserver REQUIRE_VERIFIED_MODEL_SCOPE=true. L'API refusera les autres
familles au lieu de leur appliquer silencieusement un modèle non compatible.

## Règle d'interprétation

Une prédiction ML est une estimation. Une simulation CST externe est une
validation numérique. Seule une campagne instrumentale documentée fournit une
validation expérimentale.
