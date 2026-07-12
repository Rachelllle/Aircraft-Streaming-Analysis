from io import BytesIO
from pathlib import Path
import os
import time

import joblib
import pandas as pd
import streamlit as st
from PIL import Image


def lire_config(chemin="config/application.properties"):
    config = {}
    path = Path(chemin)
    if not path.exists():
        return config

    for ligne in path.read_text(encoding="utf-8").splitlines():
        ligne = ligne.strip()
        if ligne and not ligne.startswith("#") and "=" in ligne:
            cle, valeur = ligne.split("=", 1)
            config[cle.strip()] = valeur.strip()
    return config


def compter_images(dossier):
    total = 0
    if os.path.exists(dossier):
        for racine, _, fichiers in os.walk(dossier):
            total += sum(1 for f in fichiers if f.lower().endswith(".jpg"))
    return total


def extraire_features(image_bytes, nom_image):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    largeur, hauteur = image.size
    pixels = image.load()

    hist = [0.0] * 16
    total_pixels = 0

    for y in range(0, hauteur, 4):
        for x in range(0, largeur, 4):
            r, g, b = pixels[x, y]
            gris = (r + g + b) // 3
            hist[gris // 16] += 1
            total_pixels += 1

    if total_pixels > 0:
        hist = [valeur / total_pixels for valeur in hist]

    features = {
        "image": nom_image,
        "classe": "inconnu",
        "taille": len(image_bytes),
        "moyenne_octets": sum(image_bytes) / len(image_bytes) if image_bytes else 0.0,
        "largeur": largeur,
        "hauteur": hauteur,
    }

    for index, valeur in enumerate(hist):
        features[f"hist_{index}"] = valeur

    return features


def niveau_confiance(confiance):
    if confiance >= 0.8:
        return "forte"
    if confiance >= 0.5:
        return "moyenne"
    return "faible"


def predire_image(features, model_path):
    data = joblib.load(model_path)
    model = data["model"]
    label_encoder = data["le"]

    df = pd.DataFrame([features])
    X = df.drop(columns=["image", "classe"])

    probabilites = model.predict_proba(X)[0]
    classes = label_encoder.inverse_transform(range(len(probabilites)))

    prediction_index = int(probabilites.argmax())
    prediction = classes[prediction_index]
    confiance = float(probabilites[prediction_index])

    return prediction, confiance


st.set_page_config(page_title="Streaming d'images d'avions", layout="wide")

config = lire_config()
mode = config.get("app.mode", "train")
input_path = config.get("input.path", "data/input")
dest_path = config.get("intermediate.path", "data/output")
output_path = config.get("output.path", "data/predictions")
model_path = config.get("model.path", "models/model.pkl")
modele_present = Path(model_path).exists()

if "historique_flux" not in st.session_state:
    st.session_state.historique_flux = []
if "temps_depart_flux" not in st.session_state:
    st.session_state.temps_depart_flux = time.time()

with st.sidebar:
    st.header("Parametres")
    refresh_sec = st.slider("Refresh (secondes)", min_value=2, max_value=30, value=2)
    auto_refresh = st.checkbox("Auto refresh", value=True)
    if st.button("Actualiser maintenant"):
        st.rerun()

st.title("Evolution du flux d'images d'avions")

total = compter_images(dest_path)
temps_ecoule = int(time.time() - st.session_state.temps_depart_flux)
st.session_state.historique_flux.append(
    {"temps (s)": temps_ecoule, "images traitees": total}
)

df_flux = pd.DataFrame(st.session_state.historique_flux)

col1, col2 = st.columns(2)
col1.metric("Images traitees (cumule)", total)
col2.metric("Dernier releve", f"t = {temps_ecoule}s")
st.line_chart(df_flux.set_index("temps (s)"))

st.divider()

st.title("Prediction d'une image d'avion")
st.caption("Chargez une image et lancez une prediction.")

with st.expander("Rappel du fonctionnement du projet", expanded=True):
    st.markdown(
        f"""
- `Microservice 1` lit `Input` : `{input_path}`
- il ecrit dans `Dest` : `{dest_path}`
- `Microservice 2` lit `Dest` et ecrit dans `Output` : `{output_path}`
- le script Python entraine puis exporte le modele : `{model_path}`
- ensuite, Streamlit peut predire une image

Important :

- iteration 1 : pas de `predict`
- iteration 2 : le modele est deja exporte
- le passage a la prediction reste manuel
        """
    )

col3, col4 = st.columns(2)
col3.metric("Mode courant", mode)
col4.metric("Modele disponible", "oui" if modele_present else "non")

if not modele_present:
    st.warning(
        "Le modele n'est pas encore disponible. "
        "Il faut d'abord lancer l'entrainement avec le service Python."
    )

image_chargee = st.file_uploader(
    "Choisir une image a predire",
    type=["jpg", "jpeg", "png", "bmp"],
)

if image_chargee is not None:
    image_bytes = image_chargee.getvalue()
    features = extraire_features(image_bytes, image_chargee.name)

    st.image(image_bytes, caption=image_chargee.name, use_container_width=True)

    info1, info2, info3 = st.columns(3)
    info1.metric("Largeur", features["largeur"])
    info2.metric("Hauteur", features["hauteur"])
    info3.metric("Taille fichier", f'{features["taille"]} octets')

    if st.button("Lancer la prediction", disabled=not modele_present):
        try:
            debut = time.perf_counter()
            prediction, confiance = predire_image(features, model_path)
            duree_ms = (time.perf_counter() - debut) * 1000

            st.success("Prediction terminee")
            st.write(f"**Classe predite :** {prediction}")
            st.write(f"**Confiance :** {confiance:.2%}")
            st.write(f"**Niveau de confiance :** {niveau_confiance(confiance)}")
            st.write(f"**Temps de prediction :** {duree_ms:.0f} ms")

        except Exception as exc:
            st.error(f"Erreur pendant la prediction : {exc}")
else:
    st.info("Chargez une image pour tester la prediction.")

if auto_refresh:
    time.sleep(refresh_sec)
    st.rerun()
