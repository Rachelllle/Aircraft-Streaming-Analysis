import sys

import joblib
import pandas as pd


def main():
    model_path, csv_entree, csv_sortie = sys.argv[1], sys.argv[2], sys.argv[3]

    data = joblib.load(model_path)
    model, le = data["model"], data["le"]

    df = pd.read_csv(csv_entree)
    X = df.drop(columns=["image", "classe"])

    df["classe_predite"] = le.inverse_transform(model.predict(X))
    df["confiance"] = model.predict_proba(X).max(axis=1).round(4)

    df.to_csv(csv_sortie, index=False)


if __name__ == "__main__":
    main()
