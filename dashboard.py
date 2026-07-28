from pathlib import Path
import os
import sys
import time

import joblib
import pandas as pd
import streamlit as st

sys.path.insert(0, "ml-service")
from features import extraire_features

REFRESH_SEC = 2
NIVEAUX = ["constructeur", "famille", "variante"]


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






def niveau_confiance(confiance):
    if confiance >= 0.8:
        return "forte"
    if confiance >= 0.5:
        return "moyenne"
    return "faible"


def predire_image(features, model_path):
    modeles = joblib.load(model_path)
    X = pd.DataFrame([features]).drop(columns=["image", "classe"])

    resultats = []
    for niveau in NIVEAUX:
        model, le = modeles[niveau]["model"], modeles[niveau]["le"]
        probabilites = model.predict_proba(X)[0]
        index = int(probabilites.argmax())
        classe = le.inverse_transform([index])[0]
        resultats.append((niveau, classe, float(probabilites[index])))
    return resultats




st.set_page_config(page_title="Aircraft Streaming", page_icon="✈", layout="wide")

st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}
header {background: transparent;}

.block-container {padding-top: 3rem; max-width: 1400px;}

h1 {font-weight: 700; letter-spacing: -0.02em; margin-bottom: 0.2rem;}

section[data-testid="stSidebar"] {
    border-right: 1px solid rgba(128,128,128,0.15);
}
section[data-testid="stSidebar"] h1 {
    font-size: 1.15rem; letter-spacing: 0.06em; text-transform: uppercase;
    opacity: 0.75; margin-bottom: 1.5rem;
}
section[data-testid="stSidebar"] label {
    padding: 0.35rem 0; font-size: 0.95rem;
}

div[data-testid="stMetric"] {
    background: rgba(128,128,128,0.07);
    border: 1px solid rgba(128,128,128,0.15);
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
div[data-testid="stMetricLabel"] p {
    font-size: 0.8rem; text-transform: uppercase;
    letter-spacing: 0.06em; opacity: 0.6;
}
div[data-testid="stMetricValue"] {font-weight: 600;}

div[data-testid="stFileUploader"] section {
    border: 1px dashed rgba(128,128,128,0.35);
    border-radius: 12px; background: rgba(128,128,128,0.04);
}

div[data-testid="stImage"] img {border-radius: 12px;}

.stButton button {
    border-radius: 10px; font-weight: 600; padding: 0.6rem 1rem;
}
</style>
""", unsafe_allow_html=True)

config = lire_config()
dest_path = config.get("intermediate.path", "data/output")
model_path = config.get("model.path", "models/model.pkl")
modele_present = Path(model_path).exists()

if "historique_flux" not in st.session_state:
    st.session_state.historique_flux = []
if "temps_depart_flux" not in st.session_state:
    st.session_state.temps_depart_flux = time.time()

with st.sidebar:
    st.title("Aircraft Streaming")
    page = st.radio("Navigation", ["Evolution du flux", "Prediction"], label_visibility="collapsed")


if page == "Evolution du flux":
    st.title("Evolution du flux d'images d'avions")
    st.caption("Images copiees par le Producer dans le dossier intermediaire, en temps reel.")
    st.write("")

    total = compter_images(dest_path)
    temps_ecoule = int(time.time() - st.session_state.temps_depart_flux)
    st.session_state.historique_flux.append(
        {"temps (s)": temps_ecoule, "images traitees": total}
    )

    df_flux = pd.DataFrame(st.session_state.historique_flux)

    col1, col2 = st.columns(2)
    col1.metric("Images traitees (cumule)", total)
    col2.metric("Dernier releve", f"t = {temps_ecoule}s")
    st.write("")
    st.line_chart(df_flux.set_index("temps (s)"), height=600, color="#3987e5")

    time.sleep(REFRESH_SEC)
    st.rerun()

else:
    st.title("Prediction d'une image")
    st.caption("Chargez une photo d'avion pour obtenir le constructeur predit par le modele.")
    st.write("")

    if not modele_present:
        st.warning("Le modele n'est pas encore disponible. Lancez d'abord l'entrainement Python.")

    image_chargee = st.file_uploader(
        "Choisir une image", type=["jpg", "jpeg", "png", "bmp"], label_visibility="collapsed"
    )

    if image_chargee is None:
        st.info("Chargez une image pour lancer une prediction.")
    else:
        image_bytes = image_chargee.getvalue()
        gauche, droite = st.columns([1, 1])

        with gauche:
            st.image(image_bytes, caption=image_chargee.name, width="stretch")

        with droite:
            if st.button("Lancer la prediction", disabled=not modele_present, width="stretch"):
                try:
                    features = extraire_features(image_bytes, image_chargee.name)
                    resultats = predire_image(features, model_path)

                    st.write("")
                    for niveau, classe, confiance in resultats:
                        st.metric(
                            niveau.capitalize(),
                            classe,
                            f"confiance {confiance:.1%} ({niveau_confiance(confiance)})",
                            delta_color="off",
                        )
                except Exception as exc:
                    st.error(f"Erreur pendant la prediction : {exc}")
