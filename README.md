# Aircraft-Streaming-Analysis

## Execution

### Step 1 - Scala dataset preparation

```bash
# Full preparation pipeline
sbt run

# Individual stages
sbt "run parse"   # -> data/parsing_output/parsed_dataset.parquet
sbt "run embed"   # -> data/parsing_output/raw_embeddings.parquet

# Optional scoring entrypoint
sbt "run score"
```

### Step 2 - Python training

```bash
python src/main/python/Training/training.py
```

This reads `data/parsing_output/raw_embeddings.parquet`, trains three PyTorch MLP classifiers
(`manufacturer`, `family`, `variant`), and writes them to `data/models/` as:

- `*_model.pt`
- `*_meta.pkl`

### Step 3 - Prediction UI

```bash
python src/main/python/web_interface/app.py
# -> http://localhost:5000
```

Predictions are written to `outputs/predictions.json`.

### Step 4 - Dashboard

```bash
streamlit run src/main/python/web_interface/streamlit_app.py
```

## Important note

The repository currently trains PyTorch models in Python.
The Scala prediction/scoring code still expects Spark ML `PipelineModel` directories, so
`sbt "run score"` is intentionally not supported for real inference with the current
artifacts in `data/models/`.
