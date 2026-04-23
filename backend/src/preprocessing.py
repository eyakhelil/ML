import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

def preprocess(df, target_col="G3", save_dir=None):
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed")
    os.makedirs(save_dir, exist_ok=True)
    
    df = df.copy()
    
    # Normalisation des colonnes pour éviter les KeyError dus à la casse ou aux espaces
    df.columns = [c.strip().strip("'").strip('"').lower() for c in df.columns]
    
    # Mapper les features demandées (en minuscules)
    features_to_keep = [
        "g1", "g2", "failures", "absences", "studytime", "freetime", 
        "goout", "health", "medu", "fedu", "traveltime",
        "school", "sex", "address", "famsize", "pstatus", "higher", "internet"
    ]
    
    # Vérifier la présence des colonnes
    missing = [c for c in features_to_keep if c not in df.columns]
    if missing:
        print(f"ATTENTION: Colonnes manquantes dans le dataset: {missing}")
        # On essaie de continuer avec ce qu'on a
        features_to_keep = [c for c in features_to_keep if c in df.columns]
    
    X = df[features_to_keep].copy()
    
    # S'assurer que 'pass' est calculé sur la colonne G3 normalisée
    # (qui est maintenant 'g3' car on a tout mis en minuscules)
    g3_col = "g3" if "g3" in df.columns else "G3"
    df["pass"] = (df[g3_col.lower()] >= 10).astype(int)
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