import streamlit as st
import os
import time
import pandas as pd

st.set_page_config(page_title="Streaming d'images d'avions", layout="wide")
st.title("Évolution du flux d'images d'avions")

output_dir = "data/output"
refresh_sec = 2

placeholder = st.empty()
history = []

def compter_images(dossier):
    """Compte toutes les images .jpg dans le dossier de sortie (récursif) (os.listdir si nn récursif) """
    total = 0
    if os.path.exists(dossier):
        for racine, _, fichiers in os.walk(dossier):
            total += sum(1 for f in fichiers if f.lower().endswith(".jpg"))
    return total

t = 0
while True:
    total = compter_images(output_dir)
    history.append({"temps (s)": t, "images traitées": total})
    df = pd.DataFrame(history)

    with placeholder.container():
        col1, col2 = st.columns(2)
        col1.metric("Images traitées (cumulé)", total)
        col2.metric("Dernier relevé", f"t = {t}s")
        st.line_chart(df.set_index("temps (s)"))

    t += refresh_sec
    time.sleep(refresh_sec)