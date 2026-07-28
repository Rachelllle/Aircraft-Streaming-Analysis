import sys

import joblib
import pandas as pd

NIVEAUX = ["constructeur", "famille", "variante"]


def main():
    model_path, parquet_entree, parquet_sortie = sys.argv[1], sys.argv[2], sys.argv[3]

    modeles = joblib.load(model_path)

    df = pd.read_parquet(parquet_entree)
    X = df.drop(columns=["image", "classe"])

    for niveau in NIVEAUX:
        model, le = modeles[niveau]["model"], modeles[niveau]["le"]
        df[niveau + "_predit"] = le.inverse_transform(model.predict(X))
        df["confiance_" + niveau] = model.predict_proba(X).max(axis=1).round(4)

    df.to_parquet(parquet_sortie, index=False)


if __name__ == "__main__":
    main()
