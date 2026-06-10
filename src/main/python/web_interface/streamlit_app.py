import os
import json
from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ── Config ───────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[7]
BASE_PATH   = str(_PROJECT_ROOT / "data" / "aircraft_data")
OUTPUT_PATH = str(_PROJECT_ROOT / "outputs")

st.set_page_config(page_title='Aircraft Classifier', page_icon='✈', layout='wide')


@st.cache_data
def load_data():
    return pd.read_pickle(os.path.join(BASE_PATH, 'embeddings_vit.pkl'))


def load_scores():
    path = os.path.join(OUTPUT_PATH, 'scores.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_predictions():
    path = os.path.join(OUTPUT_PATH, 'predictions.json')
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


# ── Sidebar ───────────────────────────────────────────────────────────────────
page = st.sidebar.selectbox('Navigation', [
    'Dataset Overview',
    'Model Scoring',
    'Predictions History',
])

# ════════════════════════════════
# PAGE 1 — Dataset Overview
# ════════════════════════════════
if page == 'Dataset Overview':
    st.title('Dataset Overview')
    st.markdown('**FGVC-Aircraft** — Fine-Grained Visual Classification of Aircraft')

    df = load_data()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Total Images', '10,000')
    col2.metric('Manufacturers', df['manufacturer'].nunique())
    col3.metric('Families',      df['family'].nunique())
    col4.metric('Variants',      df['variant'].nunique())

    st.markdown('---')
    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Split Distribution')
        split_counts = df['split'].value_counts()
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(split_counts, labels=split_counts.index, autopct='%1.1f%%',
               colors=['#4CAF50', '#2196F3', '#FF9800'], startangle=90)
        ax.set_title('Train / Val / Test')
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader('Top 10 Manufacturers')
        top_manuf = df['manufacturer'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(6, 5))
        top_manuf.plot(kind='barh', ax=ax, color='#2196F3')
        ax.set_xlabel('Number of images')
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown('---')
    st.subheader('Top 15 Families')
    top_family = df['family'].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(14, 5))
    top_family.plot(kind='bar', ax=ax, color='#4CAF50')
    ax.set_ylabel('Number of images')
    ax.set_title('Most represented aircraft families')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

# ════════════════════════════════
# PAGE 2 — Model Scoring
# ════════════════════════════════
elif page == 'Model Scoring':
    st.title('Model Scoring')
    st.markdown('Results from the last evaluation on the test set.')

    scores = load_scores()
    if scores is None:
        st.warning('No scores found. Run `sbt "run score"` first.')
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric('Manufacturer Accuracy', f"{scores['manufacturer']['accuracy']}%",
                    f"F1: {scores['manufacturer']['f1']}%")
        col2.metric('Family Accuracy',       f"{scores['family']['accuracy']}%",
                    f"F1: {scores['family']['f1']}%")
        col3.metric('Variant Accuracy',      f"{scores['variant']['accuracy']}%",
                    f"F1: {scores['variant']['f1']}%")

        st.markdown('---')
        labels = [
            f"Manufacturer\n({scores['manufacturer']['classes']} classes)",
            f"Family\n({scores['family']['classes']} classes)",
            f"Variant\n({scores['variant']['classes']} classes)",
        ]
        accs = [scores[k]['accuracy'] for k in ('manufacturer', 'family', 'variant')]
        f1s  = [scores[k]['f1']       for k in ('manufacturer', 'family', 'variant')]
        colors = ['#4CAF50', '#2196F3', '#FF9800']

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for bars, vals, title, ylabel, ax in [
            (axes[0].bar(labels, accs, color=colors, width=0.5), accs,
             'Accuracy per classification level', 'Accuracy (%)', axes[0]),
            (axes[1].bar(labels, f1s,  color=colors, width=0.5), f1s,
             'F1-Score per classification level', 'F1-Score (%)', axes[1]),
        ]:
            ax.set_title(title, fontsize=13, pad=12)
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, 105)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5,
                        f'{v:.1f}%', ha='center', fontweight='bold', fontsize=11)

        plt.suptitle('Aircraft Classification — ViT + Linear Probe (PyTorch)', fontsize=14, y=1.02)
        plt.tight_layout()
        st.pyplot(fig)

# ════════════════════════════════
# PAGE 3 — Predictions History
# ════════════════════════════════
elif page == 'Predictions History':
    st.title('Predictions History')
    st.markdown('All predictions made via the Flask app.')

    predictions = load_predictions()
    if not predictions:
        st.warning('No predictions yet. Use the Flask app first.')
        st.info('Run `python web_interface/app.py` then open http://localhost:5000')
    else:
        col1, col2 = st.columns(2)
        col1.metric('Total Predictions', len(predictions))
        col2.metric('Last Prediction',   predictions[-1]['timestamp'])

        st.markdown('---')
        st.subheader('Prediction Log')
        df_pred = pd.DataFrame([{
            'Timestamp':    p['timestamp'],
            'Image':        p['image'],
            'Manufacturer': f"{p['manufacturer']['label']} ({p['manufacturer']['confidence']:.1f}%)",
            'Family':       f"{p['family']['label']} ({p['family']['confidence']:.1f}%)",
            'Variant':      f"{p['variant']['label']} ({p['variant']['confidence']:.1f}%)",
        } for p in predictions])
        st.dataframe(df_pred, use_container_width=True)

        st.markdown('---')
        st.subheader('Confidence over time')
        fig, ax = plt.subplots(figsize=(14, 5))
        x = range(len(predictions))
        ax.plot(x, [p['manufacturer']['confidence'] for p in predictions],
                label='Manufacturer', color='#4CAF50', marker='o', linewidth=2)
        ax.plot(x, [p['family']['confidence'] for p in predictions],
                label='Family', color='#2196F3', marker='o', linewidth=2)
        ax.plot(x, [p['variant']['confidence'] for p in predictions],
                label='Variant', color='#FF9800', marker='o', linewidth=2)
        ax.set_xlabel('Prediction #')
        ax.set_ylabel('Confidence (%)')
        ax.set_ylim(0, 100)
        ax.set_title('Model confidence per prediction')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
