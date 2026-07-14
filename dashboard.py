from io import BytesIO
from pathlib import Path
import os
import time

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from sklearn.metrics import confusion_matrix

SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CATEGORIEL = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]


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


# repartition des classes
def figure_repartition_classes(dossier_input):
    lignes = []
    if os.path.exists(dossier_input):
        for classe in sorted(os.listdir(dossier_input)):
            chemin_classe = os.path.join(dossier_input, classe)
            if os.path.isdir(chemin_classe):
                nb_images = sum(1 for f in os.listdir(chemin_classe) if f.lower().endswith((".jpg", ".png")))
                lignes.append({"classe": classe, "nb_images": nb_images})
    df_repartition = pd.DataFrame(lignes)
    couleurs_classe = {c: CATEGORIEL[i % len(CATEGORIEL)] for i, c in enumerate(df_repartition["classe"])}

    fig = go.Figure(data=go.Bar(
        x=df_repartition["classe"],
        y=df_repartition["nb_images"],
        marker_color=[couleurs_classe[c] for c in df_repartition["classe"]],
        text=df_repartition["nb_images"],
        textposition="outside",
        textfont=dict(color="#0b0b0b", size=14),
        hovertext=[f"{row.classe} : {row.nb_images} images" for row in df_repartition.itertuples()],
        hoverinfo="text",
    ))
    fig.update_layout(
        xaxis_title="Classe",
        yaxis_title="Nombre d'images",
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        font=dict(color="#0b0b0b"),
        margin=dict(l=60, r=20, t=20, b=60),
        showlegend=False,
    )
    fig.update_yaxes(gridcolor="#e1e0d9", zerolinecolor="#c3c2b7") 
    fig.update_xaxes(showgrid=False)
    return fig


# matrice de confusion
def figure_matrice_confusion(df_test):
    classes = sorted(set(df_test["classe"]) | set(df_test["classe_predite"]))
    y_vrai = df_test["classe"]
    y_predit = df_test["classe_predite"]
    matrice = confusion_matrix(y_vrai, y_predit, labels=classes)

    total_par_ligne = matrice.sum(axis=1, keepdims=True)
    total_par_ligne[total_par_ligne == 0] = 1
    pourcentage = (matrice / total_par_ligne * 100).round(1)

    hover_text = [
        [
            f"Reel : {classes[i]}<br>Predit : {classes[j]}<br>"
            f"Nombre : {matrice[i][j]}<br>Part de la classe reelle : {pourcentage[i][j]}%"
            for j in range(len(classes))
        ]
        for i in range(len(classes))
    ]

    fig = go.Figure(data=go.Heatmap(
        z=matrice,
        x=classes,
        y=classes,
        colorscale=[[i / (len(SEQ_BLUE) - 1), c] for i, c in enumerate(SEQ_BLUE)],
        hovertext=hover_text,
        hoverinfo="text",
        showscale=True,
        colorbar=dict(title="Nombre"),
        xgap=2,
        ygap=2,
        zmin=0,
    ))

    max_val = matrice.max() if matrice.max() > 0 else 1
    annotations = []
    for i, classe_reelle in enumerate(classes):
        for j, classe_predite in enumerate(classes):
            valeur = matrice[i][j]
            couleur_texte = "#ffffff" if valeur > max_val * 0.55 else "#0b0b0b"
            annotations.append(dict(
                x=classe_predite, y=classe_reelle, text=str(valeur),
                showarrow=False, font=dict(color=couleur_texte, size=16),
            ))

    fig.update_layout(
        annotations=annotations,
        xaxis_title="Classe predite",
        yaxis_title="Classe reelle",
        yaxis_autorange="reversed",
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        font=dict(color="#0b0b0b"),
        margin=dict(l=60, r=20, t=20, b=60),
    )
    return fig


# accuracy par classe
def figure_accuracy_par_classe(df_test):
    classes = sorted(df_test["classe"].unique())
    couleurs_classe = {c: CATEGORIEL[i % len(CATEGORIEL)] for i, c in enumerate(classes)}

    lignes = []
    for c in classes:
        sous = df_test[df_test["classe"] == c]
        total = len(sous)
        corrects = int((sous["classe_predite"] == c).sum())
        lignes.append({
            "classe": c,
            "accuracy": corrects / total if total else 0.0,
            "corrects": corrects,
            "total": total,
        })
    df_acc = pd.DataFrame(lignes)

    fig = go.Figure(data=go.Bar(
        x=df_acc["classe"],
        y=df_acc["accuracy"],
        marker_color=[couleurs_classe[c] for c in df_acc["classe"]],
        text=[f"{a:.0%}" for a in df_acc["accuracy"]],
        textposition="outside",
        textfont=dict(color="#0b0b0b", size=14),
        hovertext=[
            f"{row.classe}<br>{row.corrects}/{row.total} correctement predites ({row.accuracy:.1%})"
            for row in df_acc.itertuples()
        ],
        hoverinfo="text",
    ))
    fig.update_layout(
        xaxis_title="Classe reelle",
        yaxis_title="Accuracy",
        yaxis_tickformat=".0%",
        yaxis_range=[0, 1.15],
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        font=dict(color="#0b0b0b"),
        margin=dict(l=60, r=20, t=20, b=60),
        showlegend=False,
    )
    fig.update_yaxes(gridcolor="#e1e0d9", zerolinecolor="#c3c2b7")
    fig.update_xaxes(showgrid=False)
    return fig


# debit par micro-batch d Producer
def figure_debit_batchs(df_stats):
    fig = go.Figure(data=go.Bar(
        x=df_stats["batchId"],
        y=df_stats["debitImagesParSeconde"],
        marker_color="#2a78d6",
        text=[f"{d:.0f}" for d in df_stats["debitImagesParSeconde"]],
        textposition="outside",
        textfont=dict(color="#0b0b0b", size=13),
        hovertext=[
            f"Batch {row.batchId}<br>{row.nbImages} images en {row.dureeSecondes:.2f}s"
            f"<br>{row.debitImagesParSeconde:.1f} images/s"
            for row in df_stats.itertuples()
        ],
        hoverinfo="text",
    ))
    fig.update_layout(
        xaxis_title="Micro-batch",
        yaxis_title="Debit (images/s)",
        xaxis=dict(dtick=1),
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        font=dict(color="#0b0b0b"),
        margin=dict(l=60, r=20, t=20, b=60),
        showlegend=False,
    )
    fig.update_yaxes(gridcolor="#e1e0d9", zerolinecolor="#c3c2b7")
    fig.update_xaxes(showgrid=False)
    return fig


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

# evolution du flux d'images
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

# debit par micro-batch d Producer
producer_stats_path = Path("data/producer_stats.csv")
if producer_stats_path.exists():
    st.subheader("Debit par micro-batch (Producer)")
    df_stats = pd.read_csv(producer_stats_path)
    st.plotly_chart(figure_debit_batchs(df_stats), width="stretch", theme=None)

st.divider()

# dataset
st.title("Dataset")
st.caption("Composition du jeu de donnees utilisé ")

# repartition des classes
st.subheader("Repartition des classes")
st.plotly_chart(figure_repartition_classes(input_path), width="stretch", theme=None)

st.divider()

# resultats du modele
st.title("Resultats du modele")

predictions_test_path = Path("data/predictions_test.csv")
if predictions_test_path.exists():
    df_test = pd.read_csv(predictions_test_path)
    accuracy_test = (df_test["classe"] == df_test["classe_predite"]).mean()
    st.metric("Accuracy de", f"{accuracy_test:.1%}")

    # matrice de confusion
    st.plotly_chart(figure_matrice_confusion(df_test), width="stretch", theme=None)

    # accuracy par classe
    st.subheader("Accuracy par classe")
    st.plotly_chart(figure_accuracy_par_classe(df_test), width="stretch", theme=None)
else:
    st.info(
        "Pas encore de predictions de test disponibles. "
        "Lancez l'entrainement (ml-service/train.py) pour generer la matrice de confusion."
    )

st.divider()

# prediction d'une image
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
