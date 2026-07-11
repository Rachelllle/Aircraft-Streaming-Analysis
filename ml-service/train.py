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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Accuracy : {accuracy:.4f}")

    os.makedirs(os.path.dirname(config["model.path"]), exist_ok=True)
    joblib.dump({"model": model, "le": le}, config["model.path"])
    print(f"Modele exporte : {config['model.path']}")


if __name__ == "__main__":
    main()
