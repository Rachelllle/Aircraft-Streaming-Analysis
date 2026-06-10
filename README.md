# Aircraft-Streaming-Analysis

## Exécution

### Étape 1 — Parsing Scala (annotations + extraction des pixels)

```bash
# Pipeline complet (parse + embed)
sbt run

# Étapes individuelles
sbt "run parse"   # annotations → data/parsing_output/parsed_dataset.parquet
sbt "run embed"   # vecteurs de pixels → data/parsing_output/raw_embeddings.parquet
```

### Étape 2 — Entraînement Python (classifieurs depuis les parquets)

```bash
python src/main/python/Training/training.py
```

Lit `data/parsing_output/raw_embeddings.parquet`, entraîne trois classifieurs MLP PyTorch (fabricant / famille / variante) sur des features de pixels 64×64×3 = 12 288 dimensions, et sauvegarde les modèles dans `data/models/`.

### Étape 3 — Interface web

```bash
# Classifieur Flask (uploader une image → obtenir des prédictions)
python src/main/python/web_interface/app.py
# → http://localhost:5000

# Tableau de bord Streamlit (stats du jeu de données + scores des modèles)
streamlit run src/main/python/web_interface/streamlit_app.py
```