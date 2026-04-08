import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

def preprocess(df: pd.DataFrame, target_col: str = "G3", save_dir: str = "data/processed"):
    """
    Prétraite le dataset :
    - Encode les variables catégorielles
    - Crée une cible binaire (pass/fail)
    - Normalise les features numériques
    - Sauvegarde les données traitées
    """
    os.makedirs(save_dir, exist_ok=True)
    
    df = df.copy()
    
    # Créer la cible binaire
    df["pass"] = (df["G3"] >= 10).astype(int)
    
    # Séparer features et cible (on exclut G1, G2, G3 pour éviter la fuite de données)
    # Pour tester l'impact de l'assiduité, notes intermédiaires et participation
    features_to_keep = [
        "absences", "studytime", "failures", "goout", "Dalc", "Walc",
        "health", "famrel", "freetime", "G1", "G2",
        "school", "sex", "address", "famsize", "Pstatus",
        "Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian",
        "traveltime", "schoolsup", "famsup", "paid", "activities",
        "nursery", "higher", "internet", "romantic"
    ]
    
    X = df[features_to_keep].copy()
    y = df["pass"]
    
    # Encoder les colonnes catégorielles
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le
    
    # Sauvegarde des encodeurs
    joblib.dump(encoders, os.path.join(save_dir, "encoders.pkl"))
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Sauvegarde du scaler
    joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))
    
    # Sauvegarde des données
    pd.DataFrame(X_train_scaled, columns=X.columns).to_csv(
        os.path.join(save_dir, "X_train.csv"), index=False
    )
    pd.DataFrame(X_test_scaled, columns=X.columns).to_csv(
        os.path.join(save_dir, "X_test.csv"), index=False
    )
    y_train.to_csv(os.path.join(save_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(save_dir, "y_test.csv"), index=False)
    
    print(f"Prétraitement terminé. Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, list(X.columns)

if __name__ == "__main__":
    from data_loader import load_data
    df = load_data("../data/raw/student-mat.csv")
    preprocess(df, save_dir="../data/processed")