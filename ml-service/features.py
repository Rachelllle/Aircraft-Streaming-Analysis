from io import BytesIO

from PIL import Image


def extraire_features(image_bytes, nom_image=None):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    largeur, hauteur = image.size

    valeurs = {
        "taille": len(image_bytes),
        "moyenne_octets": sum(image_bytes) / len(image_bytes) if image_bytes else 0.0,
        "largeur": largeur,
        "hauteur": hauteur,
        "ratio": largeur / hauteur if hauteur else 0.0,
    }

    if nom_image is not None:
        valeurs = {"image": nom_image, "classe": "inconnu", **valeurs}
    return valeurs
