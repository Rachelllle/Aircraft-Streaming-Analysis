import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Configuration des chemins et logger
from src.logger_config import logger
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.config import BASE_PATH, MODEL_PATH

def train_random_forest(df_train, df_test, target_col, model_name):
    """
    Entraîne un modèle Random Forest pour une cible spécifique (family, manufacturer ou variant)
    en utilisant les embeddings bruts (768D).
    """
    # Extraction des features X (Convertit la liste d'embeddings en matrice NumPy)
    X_train = np.array(df_train['embedding'].tolist(), dtype=np.float32)
    X_test = np.array(df_test['embedding'].tolist(), dtype=np.float32)
    
    # Encodage des labels textuels (ex: 'Boeing', 'Airbus' -> 0, 1)
    le = LabelEncoder()
    y_train = le.fit_transform(df_train[target_col].values)
    y_test = le.transform(df_test[target_col].values)
    
    num_classes = len(le.classes_)
    logger.info(f"Entraînement de Random Forest pour {model_name} ({num_classes} classes, {len(X_train)} échantillons)...")
    
    # Initialisation du modèle Random Forest
    # n_jobs=-1 permet d'utiliser tous les cœurs CPU pour accélérer le calcul
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    # Entraînement
    rf_model.fit(X_train, y_train)
    
    # Évaluation sur le jeu de test
    y_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"  [Résultat] Test Accuracy pour {model_name} : {acc:.4f}")
    
    # Sauvegarde des artefacts (Le modèle RF et l'encodeur de labels)
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    # Sauvegarde commune des métadonnées et du modèle via joblib
    model_data = {
        'model': rf_model,
        'le': le,
        'num_classes': num_classes,
        'accuracy': acc
    }
    joblib.dump(model_data, f"{MODEL_PATH}/rf_{model_name}_model.pkl")
    logger.info(f"Modèle {model_name} sauvegardé sous : {MODEL_PATH}/rf_{model_name}_model.pkl\n")
    
    return rf_model, le

def train_all_models():
    logger.info("Chargement des embeddings bruts depuis le fichier Pickle...")
    df = pd.read_pickle(BASE_PATH + '/embeddings_vit.pkl')

    # Utilisation de Train + Val pour l'entraînement (comme dans votre script PyTorch original)
    df_train = df[df['split'].isin(['train', 'val'])].reset_index(drop=True)
    df_test  = df[df['split'] == 'test'].reset_index(drop=True)

    logger.info(f"Données chargées — Train/Val: {len(df_train)} images | Test: {len(df_test)} images")

    # Entraînement des 3 niveaux de classification requis par FGVC-Aircraft
    train_random_forest(df_train, df_test, 'manufacturer', 'manufacturer')
    train_random_forest(df_train, df_test, 'family',       'family')
    train_random_forest(df_train, df_test, 'variant',      'variant')

if __name__ == '__main__':
    train_all_models()
    logger.info("Tous les modèles Random Forest ont été entraînés et sauvegardés avec succès !")