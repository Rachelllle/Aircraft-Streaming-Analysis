import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder

# ── Config ────────────────────────────────────────────────────────────────────
_PROJECT_ROOT  = Path(__file__).resolve().parents[4]
PARSING_OUTPUT = str(_PROJECT_ROOT / "data" / "parsing_output")
MODEL_PATH     = str(_PROJECT_ROOT / "data" / "models")
os.makedirs(MODEL_PATH, exist_ok=True)

FEATURE_DIM = 64 * 64 * 3  # 12 288 — produit par Scala Embeddings

# ── Logger ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("AircraftTraining")


def load_embeddings() -> pd.DataFrame:
    """Lit raw_embeddings.parquet produit par Scala.
    Le DenseVector Spark est stocke comme struct {type, size, indices, values}.
    """
    path = os.path.join(PARSING_OUTPUT, "raw_embeddings.parquet")
    logger.info(f"Lecture de {path}")
    df = pd.read_parquet(path)
    df["embedding"] = df["features_raw"].apply(
        lambda v: np.array(v["values"], dtype=np.float32)
    )
    return df[["embedding", "manufacturer", "family", "variant", "split"]]


def train_model(df: pd.DataFrame, target_col: str, model_name: str):
    X = np.stack(df["embedding"].values).astype(np.float32)  # (N, 12288)

    le = LabelEncoder()
    y  = le.fit_transform(df[target_col].values)
    num_classes = len(le.classes_)
    logger.info(f"Entrainement {model_name} -- {num_classes} classes, {len(X)} exemples")

    mean = X.mean(axis=0)
    std  = X.std(axis=0) + 1e-8
    X    = (X - mean) / std

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader = DataLoader(
        TensorDataset(torch.tensor(X), torch.tensor(y, dtype=torch.long)),
        batch_size=64, shuffle=True, num_workers=0,
    )

    model = nn.Sequential(
        nn.Linear(FEATURE_DIM, 1024), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(1024, 512),         nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(50):
        model.train()
        total_loss = 0.0
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            logger.info(f"  epoch {epoch+1}/50 -- loss: {total_loss/len(loader):.4f}")

    joblib.dump(
        {
            "le":          le,
            "num_classes": num_classes,
            "mean":        mean,
            "std":         std,
            "feature_dim": FEATURE_DIM,
            "model_type":  "mlp",
        },
        os.path.join(MODEL_PATH, f"{model_name}_meta.pkl"),
    )
    torch.save(model.state_dict(), os.path.join(MODEL_PATH, f"{model_name}_model.pt"))
    logger.info(f"{model_name} sauvegarde -> {MODEL_PATH}/")
    return model, le, mean, std


def train_all_models():
    df = load_embeddings()
    df_train = df[df["split"].isin(["train", "val"])].reset_index(drop=True)
    df_test  = df[df["split"] == "test"].reset_index(drop=True)
    logger.info(f"train+val : {len(df_train)} | test : {len(df_test)}")

    train_model(df_train, "manufacturer", "manufacturer")
    train_model(df_train, "family",       "family")
    train_model(df_train, "variant",      "variant")


if __name__ == "__main__":
    train_all_models()
    logger.info("Tous les modeles entraines et sauvegardes.")
