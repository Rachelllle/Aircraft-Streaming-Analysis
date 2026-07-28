import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from features import extraire_features

NIVEAUX = ["constructeur", "famille", "variante"]


def lire_config(chemin="config/application.properties"):
    config = {}
    with open(chemin, encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if ligne and not ligne.startswith("#") and "=" in ligne:
                cle, valeur = ligne.split("=", 1)
                config[cle.strip()] = valeur.strip()
    return config


def lire_images(dossier):
    lignes = []
    for racine, _, fichiers in os.walk(dossier):
        for nom in sorted(fichiers):
            if not nom.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            with open(os.path.join(racine, nom), "rb") as f:
                lignes.append(extraire_features(f.read(), nom))
            if len(lignes) % 100 == 0:
                print(f"  {len(lignes)} images analysees...")
    return pd.DataFrame(lignes)


def entrainer(X, classes, niveau):
    garder = classes.map(classes.value_counts()) >= 2
    X, classes = X[garder], classes[garder]

    le = LabelEncoder()
    y = le.fit_transform(classes)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"  {niveau:13s} : {classes.nunique():3d} classes  ->  accuracy {accuracy:.4f}")

    return {"model": model, "le": le, "accuracy": accuracy}


def main():
    config = lire_config()

    print(f"Lecture des images d'entrainement ({config['train.path']})")
    df = lire_images(config["train.path"])

    labels = pd.read_csv(config["labels.path"])
    df = df.merge(labels, on="image", how="inner")
    print(f"{len(df)} images pretes")

    X = df.drop(columns=["image", "classe"] + NIVEAUX)
    modeles = {niveau: entrainer(X, df[niveau], niveau) for niveau in NIVEAUX}

    os.makedirs(os.path.dirname(config["model.path"]), exist_ok=True)
    joblib.dump(modeles, config["model.path"])
    print(f"Modeles exportes : {config['model.path']}")


if __name__ == "__main__":
    main()
