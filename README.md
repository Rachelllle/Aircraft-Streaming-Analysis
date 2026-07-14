# Aircraft-Streaming-Analysis

Classification d'images d'avions en Spark Structured Streaming.

## Architecture

```
data/input (images triees par classe)
   -> Producer (simule un flux, copie les images par batchs)
data/output
   -> Consumer (Structured Streaming, extrait les features par micro-batch)
   -> ml-service/train.py (entrainement RandomForest, hors-ligne)
   -> ml-service/predict.py (prediction appelee par le Consumer en mode "predict")
data/features / data/predictions_test.csv
   -> dashboard.py (Streamlit : flux, dataset, resultats du modele, prediction manuelle)
```

- **Producer** (`src/main/scala/Producer.scala`) : lit les images de `data/input` et les copie vers `data/output` par batchs, avec une pause entre chaque batch pour simuler un flux continu.
- **Consumer** (`src/main/scala/Consumer.scala`) : lit `data/output` en Structured Streaming, extrait des features par image (taille, moyenne d'octets, histogramme de gris), puis selon le mode :
  - `train` : exporte les features en CSV pour l'entrainement,
  - `predict` : appelle `ml-service/predict.py` pour classifier chaque micro-batch a la volee.
- **ml-service** (`ml-service/train.py`, `ml-service/predict.py`) : entrainement et prediction avec un `RandomForestClassifier` (scikit-learn).
- **dashboard.py** : interface Streamlit pour visualiser le flux, le dataset et les resultats du modele (matrice de confusion, accuracy par classe).

## Lancer le pipeline

1. Placer des images dans `data/input/<classe>/*.jpg`
2. Mettre `app.mode=train` dans `config/application.properties`, puis :
   ```bash
   sbt "runMain Producer"
   sbt "runMain Consumer"
   ```
3. Entrainer le modele :
   ```bash
   python3 ml-service/train.py
   ```
4. Repasser `app.mode=predict` dans la config, relancer le Consumer pour la prediction a la volee.
5. Lancer le dashboard :
   ```bash
   streamlit run dashboard.py
   ```
