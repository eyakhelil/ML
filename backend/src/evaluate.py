import pandas as pd
import numpy as np
import joblib
import os

def predict_student(model_path, scaler_path, encoders_path, student_data):
    model    = joblib.load(model_path)
    scaler   = joblib.load(scaler_path)
    encoders = joblib.load(encoders_path)

    # Normaliser les données (clés en minuscules, mapping yes/no -> 1/0)
    student_data_clean = {}
    for k, v in student_data.items():
        val = v
        if isinstance(v, str):
            v_low = v.lower().strip()
            if v_low == "yes": val = 1
            elif v_low == "no": val = 0
        student_data_clean[k.lower().strip()] = val

    df = pd.DataFrame([student_data_clean])

    # Encoder les colonnes catégorielles
    for col, le in encoders.items():
        if col in df.columns:
            try:
                # S'assurer que la valeur est une string pour le LabelEncoder
                # (au cas où "yes" aurait déjà été mappé en 1 mais l'encodeur l'attend)
                # Note: si c'est déjà 1, le transform échouera car il attend 'yes'
                df[col] = le.transform(df[col].astype(str))
            except Exception:
                df[col] = 0

    # Garder seulement les colonnes connues du scaler
    feature_names = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else None
    
    if feature_names is not None:
        # Ajouter les colonnes manquantes avec 0
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_names]

    X_scaled = scaler.transform(df)

    pred  = model.predict(X_scaled)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)[0][1]

    # Règle métier : Moyenne (G1, G2) < 10 => Échec
    if (student_data.get("G1", 0) + student_data.get("G2", 0)) / 2 < 10:
        pred = 0

    return {
        "prediction": int(pred),
        "label": "Succès (≥10)" if pred == 1 else "Échec (<10)",
        "probability_success": round(float(proba), 4) if proba is not None else None
    }