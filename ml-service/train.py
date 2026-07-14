import glob
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder



def lire_config(chemin="config/application.properties"):
    config = {}
    with open(chemin, encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if ligne and not ligne.startswith("#") and "=" in ligne:
                cle, valeur = ligne.split("=", 1)
                config[cle.strip()] = valeur.strip()
    return config


def main():
    config = lire_config()

    fichiers = glob.glob(os.path.join(config["output.path"], "*.csv"))
    df = pd.concat([pd.read_csv(f) for f in fichiers], ignore_index=True)
    df = df[df["classe"] != "inconnu"]
    print(f"{len(df)} images, {df['classe'].nunique()} classes")

    le = LabelEncoder()
    X = df.drop(columns=["image", "classe"])
    y = le.fit_transform(df["classe"])

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    # probas_test = model.predict_proba(X_test)
    probas_test = model.predict_proba(X_test)
    confiance_test = probas_test.max(axis=1)
    accuracy = accuracy_score(y_test, y_pred_test)
    print(f"Accuracy : {accuracy:.4f}")

    os.makedirs(os.path.dirname(config["model.path"]), exist_ok=True)
    joblib.dump({"model": model, "le": le}, config["model.path"])
    print(f"Modele exporte : {config['model.path']}")

    # ancienne version sans la confiance jai ajouté idx pour recup les noms 
    # resultats_test = pd.DataFrame({
    #     "image": df.loc[idx_test, "image"].values,
    #     "classe": le.inverse_transform(y_test),
    #     "classe_predite": le.inverse_transform(y_pred_test),
    # })
    resultats_test = pd.DataFrame({
        "image": df.loc[idx_test, "image"].values,
        "classe": le.inverse_transform(y_test),
        "classe_predite": le.inverse_transform(y_pred_test),
        "confiance": confiance_test.round(4),
    })
    resultats_test.to_csv("data/predictions_test.csv", index=False)
    print("Predictions du jeu de test exportees : data/predictions_test.csv")

    importance_features = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance_features.to_csv("data/feature_importance.csv", index=False)
    print("Feature importance exportee : data/feature_importance.csv")


if __name__ == "__main__":
    main()
