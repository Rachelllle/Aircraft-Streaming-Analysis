import os
import io
import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from flask import Flask, request, jsonify, render_template_string

# ── Config ────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_PATH   = str(_PROJECT_ROOT / "outputs")
MODEL_PATH    = str(_PROJECT_ROOT / "data" / "models")
os.makedirs(OUTPUT_PATH, exist_ok=True)

FEATURE_DIM = 64 * 64 * 3  # 12 288

# ── Logger ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("AircraftWebApp")


# ── Inférence MLP ─────────────────────────────────────────────────────────────
def _extract_pixel_features(image_path: str) -> np.ndarray:
    """Reproduit Scala Embeddings.extractPixelFeatures : redim 64x64, aplatit RGB."""
    arr = np.array(
        Image.open(image_path).convert("RGB").resize((64, 64)),
        dtype=np.float32,
    ) / 255.0
    return arr.flatten()  # (12288,)


def _build_mlp(num_classes: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(FEATURE_DIM, 1024), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(1024, 512),         nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )


def _predict(image_path: str) -> dict:
    emb    = _extract_pixel_features(image_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}
    for name in ["manufacturer", "family", "variant"]:
        meta = joblib.load(os.path.join(MODEL_PATH, f"{name}_meta.pkl"))
        le, n, mean, std = meta["le"], meta["num_classes"], meta["mean"], meta["std"]

        model = _build_mlp(n).to(device)
        model.load_state_dict(
            torch.load(os.path.join(MODEL_PATH, f"{name}_model.pt"),
                       map_location=device, weights_only=True)
        )
        model.eval()

        x = torch.tensor((emb - mean) / std, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1).cpu().numpy().squeeze()

        idx  = int(probs.argmax())
        top3 = probs.argsort()[::-1][:3]
        results[name] = {
            "label":      le.inverse_transform([idx])[0],
            "confidence": float(probs[idx] * 100),
            "top3": [
                {"label": le.inverse_transform([i])[0], "confidence": float(probs[i] * 100)}
                for i in top3
            ],
        }
    return results


def _save_prediction(image_name: str, results: dict):
    path    = os.path.join(OUTPUT_PATH, "predictions.json")
    history = json.load(open(path)) if os.path.exists(path) else []
    history.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image":     image_name,
        **{k: {"label": v["label"], "confidence": round(v["confidence"], 2)}
           for k, v in results.items()},
    })
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Prediction sauvegardee -> {path}")


# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>Aircraft Classifier</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: Arial, sans-serif; background: #0f172a; color: white; min-height: 100vh; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 20px; }
        h1 { font-size: 2rem; margin-bottom: 8px; }
        p.sub { color: #94a3b8; margin-bottom: 40px; }
        .card { background: #1e293b; border-radius: 16px; padding: 32px; width: 100%; max-width: 600px; }
        .upload-zone { border: 2px dashed #334155; border-radius: 12px; padding: 40px; text-align: center; cursor: pointer; transition: all 0.2s; margin-bottom: 20px; }
        .upload-zone:hover { border-color: #3b82f6; background: #1e3a5f; }
        .upload-zone input { display: none; }
        .upload-zone label { cursor: pointer; color: #94a3b8; font-size: 1rem; }
        .upload-zone label span { color: #3b82f6; font-weight: bold; }
        #preview { width: 100%; border-radius: 8px; margin-bottom: 20px; display: none; max-height: 300px; object-fit: contain; }
        button { width: 100%; padding: 14px; background: #3b82f6; color: white; border: none; border-radius: 10px; font-size: 1rem; cursor: pointer; font-weight: bold; transition: background 0.2s; }
        button:hover { background: #2563eb; }
        button:disabled { background: #334155; cursor: not-allowed; }
        .results { margin-top: 24px; display: none; }
        .result-item { background: #0f172a; border-radius: 10px; padding: 16px; margin-bottom: 12px; }
        .result-item .level { color: #94a3b8; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
        .result-item .label { font-size: 1.3rem; font-weight: bold; margin-bottom: 8px; }
        .result-item .confidence { color: #94a3b8; font-size: 0.9rem; margin-bottom: 10px; }
        .bar-bg { background: #1e293b; border-radius: 999px; height: 8px; }
        .bar-fill { height: 8px; border-radius: 999px; background: #3b82f6; transition: width 0.5s; }
        .top3 { margin-top: 10px; }
        .top3-item { display: flex; justify-content: space-between; font-size: 0.85rem; color: #64748b; margin-bottom: 2px; }
        .loading { text-align: center; color: #94a3b8; padding: 20px; display: none; }
        .spinner { border: 3px solid #334155; border-top: 3px solid #3b82f6; border-radius: 50%; width: 30px; height: 30px; animation: spin 0.8s linear infinite; margin: 0 auto 10px; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .manufacturer .bar-fill { background: #22c55e; }
        .family      .bar-fill { background: #3b82f6; }
        .variant     .bar-fill { background: #f97316; }
        .error-box { background: #450a0a; border: 1px solid #7f1d1d; border-radius: 8px; padding: 12px; color: #fca5a5; margin-top: 16px; display: none; }
    </style>
</head>
<body>
    <h1>Aircraft Classifier</h1>
    <p class="sub">Chargez une photo d'avion pour l'identifier</p>
    <div class="card">
        <div class="upload-zone" onclick="document.getElementById('fileInput').click()">
            <input type="file" id="fileInput" accept="image/*" onchange="previewImage(event)">
            <label>Deposez une image ou <span>parcourez</span></label>
        </div>
        <img id="preview" src="" alt="preview">
        <button id="predictBtn" onclick="predict()" disabled>Predire</button>
        <div class="loading" id="loading"><div class="spinner"></div>Analyse en cours...</div>
        <div class="error-box" id="errorBox"></div>
        <div class="results" id="results">
            <div class="result-item manufacturer">
                <div class="level">Fabricant</div>
                <div class="label" id="manuf-label"></div>
                <div class="confidence" id="manuf-conf"></div>
                <div class="bar-bg"><div class="bar-fill" id="manuf-bar" style="width:0%"></div></div>
                <div class="top3" id="manuf-top3"></div>
            </div>
            <div class="result-item family">
                <div class="level">Famille</div>
                <div class="label" id="family-label"></div>
                <div class="confidence" id="family-conf"></div>
                <div class="bar-bg"><div class="bar-fill" id="family-bar" style="width:0%"></div></div>
                <div class="top3" id="family-top3"></div>
            </div>
            <div class="result-item variant">
                <div class="level">Variante</div>
                <div class="label" id="variant-label"></div>
                <div class="confidence" id="variant-conf"></div>
                <div class="bar-bg"><div class="bar-fill" id="variant-bar" style="width:0%"></div></div>
                <div class="top3" id="variant-top3"></div>
            </div>
        </div>
    </div>
    <script>
        let selectedFile = null;
        function previewImage(event) {
            selectedFile = event.target.files[0];
            if (!selectedFile) return;
            const reader = new FileReader();
            reader.onload = e => {
                const preview = document.getElementById('preview');
                preview.src = e.target.result;
                preview.style.display = 'block';
                document.getElementById('predictBtn').disabled = false;
                document.getElementById('results').style.display = 'none';
                document.getElementById('errorBox').style.display = 'none';
            };
            reader.readAsDataURL(selectedFile);
        }
        async function predict() {
            if (!selectedFile) return;
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').style.display = 'none';
            document.getElementById('errorBox').style.display = 'none';
            document.getElementById('predictBtn').disabled = true;
            const formData = new FormData();
            formData.append('image', selectedFile);
            try {
                const response = await fetch('/predict', { method: 'POST', body: formData });
                const data = await response.json();
                if (data.error) {
                    document.getElementById('errorBox').textContent = 'Erreur : ' + data.error;
                    document.getElementById('errorBox').style.display = 'block';
                    return;
                }
                ['manufacturer', 'family', 'variant'].forEach(level => {
                    const r = data[level];
                    const prefix = level === 'manufacturer' ? 'manuf' : level;
                    document.getElementById(`${prefix}-label`).textContent = r.label;
                    document.getElementById(`${prefix}-conf`).textContent = `Confiance : ${r.confidence.toFixed(1)}%`;
                    document.getElementById(`${prefix}-bar`).style.width = Math.min(r.confidence, 100) + '%';
                    document.getElementById(`${prefix}-top3`).innerHTML = r.top3.map(t =>
                        `<div class="top3-item"><span>${t.label}</span><span>${t.confidence.toFixed(1)}%</span></div>`
                    ).join('');
                });
                document.getElementById('results').style.display = 'block';
            } catch(e) {
                document.getElementById('errorBox').textContent = 'Erreur reseau : ' + e.message;
                document.getElementById('errorBox').style.display = 'block';
            } finally {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('predictBtn').disabled = false;
            }
        }
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/predict', methods=['POST'])
def predict_route():
    if 'image' not in request.files or not request.files['image'].filename:
        return jsonify({'error': 'fichier image manquant'}), 400
    try:
        file      = request.files['image']
        temp_path = os.path.join(OUTPUT_PATH, "temp_predict.jpg")
        Image.open(io.BytesIO(file.read())).convert('RGB').save(temp_path)
        results = _predict(temp_path)
        _save_prediction(file.filename, results)
        return jsonify(results)
    except Exception as exc:
        logger.exception("Erreur lors de la prediction")
        return jsonify({'error': str(exc)}), 500


if __name__ == '__main__':
    app.run(debug=False, port=5000)
