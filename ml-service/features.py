from io import BytesIO

import numpy as np
from PIL import Image

GRID_SIZE = 8
FEATURES_PAR_CELLULE = 6
NB_FEATURES = GRID_SIZE * GRID_SIZE * FEATURES_PAR_CELLULE
SEUIL_CONTOUR = 25
PAS = 2

COLONNES = (
    ["taille", "moyenne_pixels", "densite_contours", "largeur", "hauteur", "ratio"]
    + [f"grille_{i}" for i in range(NB_FEATURES)]
)


def extraire_features(image_bytes, nom_image=None):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    largeur, hauteur = image.size
    arr = np.asarray(image, dtype=np.float64)

    cell_w = max(1, largeur // GRID_SIZE)
    cell_h = max(1, hauteur // GRID_SIZE)
    feats = np.zeros(NB_FEATURES)

    somme_globale = 0.0
    n_global = 0
    contours_global = 0

    for cy in range(GRID_SIZE):
        for cx in range(GRID_SIZE):
            xs, ys = cx * cell_w, cy * cell_h
            xe = largeur if cx == GRID_SIZE - 1 else xs + cell_w
            ye = hauteur if cy == GRID_SIZE - 1 else ys + cell_h

            bloc = arr[ys:ye:PAS, xs:xe:PAS]
            if bloc.size == 0:
                continue

            gris = bloc.sum(axis=2) / 3.0
            n = gris.size

            moy = gris.mean()
            ecart = np.sqrt(max(0.0, (gris * gris).mean() - moy * moy))
            contours = int((np.abs(np.diff(gris, axis=1)) > SEUIL_CONTOUR).sum())

            idx = (cy * GRID_SIZE + cx) * FEATURES_PAR_CELLULE
            feats[idx] = moy
            feats[idx + 1] = ecart
            feats[idx + 2] = bloc[:, :, 0].mean()
            feats[idx + 3] = bloc[:, :, 1].mean()
            feats[idx + 4] = bloc[:, :, 2].mean()
            feats[idx + 5] = contours / n

            somme_globale += gris.sum()
            n_global += n
            contours_global += contours

    valeurs = {
        "taille": len(image_bytes),
        "moyenne_pixels": somme_globale / n_global if n_global else 0.0,
        "densite_contours": contours_global / n_global if n_global else 0.0,
        "largeur": largeur,
        "hauteur": hauteur,
        "ratio": largeur / hauteur if hauteur else 0.0,
    }
    for i, valeur in enumerate(feats):
        valeurs[f"grille_{i}"] = valeur

    if nom_image is not None:
        valeurs = {"image": nom_image, "classe": "inconnu", **valeurs}
    return valeurs
