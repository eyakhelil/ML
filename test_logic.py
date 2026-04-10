import joblib
import pandas as pd
import numpy as np
import os

def test_prediction():
    base_dir = r"c:\Users\Administrator\project_studentPerformance"
    models_dir = os.path.join(base_dir, "backend", "models")
    scaler_path = os.path.join(base_dir, "backend", "data", "processed", "scaler.pkl")
    encoders_path = os.path.join(base_dir, "backend", "data", "processed", "encoders.pkl")
    
    if not os.path.exists(scaler_path):
        print("Scaler not found. Please train models first.")
        return

    scaler = joblib.load(scaler_path)
    encoders = joblib.load(encoders_path)
    
    # Un étudiant qui devrait échouer (G1=6, G2=7)
    student = {
        'absences': 4, 'studytime': 2, 'failures': 0, 'goout': 3, 
        'Dalc': 1, 'Walc': 2, 'health': 3, 'famrel': 4, 'freetime': 3, 
        'G1': 6, 'G2': 7, 
        'Medu': 2, 'Fedu': 2, 'traveltime': 2, 'school': 'GP', 
        'sex': 'M', 'address': 'U', 'famsize': 'GT3', 'Pstatus': 'T', 
        'Mjob': 'other', 'Fjob': 'other', 'reason': 'course', 
        'guardian': 'mother', 'schoolsup': 'no', 'famsup': 'yes', 
        'paid': 'no', 'activities': 'no', 'nursery': 'yes', 
        'higher': 'yes', 'internet': 'yes', 'romantic': 'no'
    }
    
    df = pd.DataFrame([student])
    for col, le in encoders.items():
        df[col] = le.transform(df[col])
        
    df = df[scaler.feature_names_in_]
    X_scaled = scaler.transform(df)
    
    print("--- Test des modèles ---")
    relevant_models = ["KNN_k3.pkl", "RF_100trees_depthNone.pkl", "LR_C1.0_lbfgs.pkl"]
    for m_name in relevant_models:
        path = os.path.join(models_dir, m_name)
        if os.path.exists(path):
            m = joblib.load(path)
            pred = m.predict(X_scaled)[0]
            label = "SUCCÈS" if pred == 1 else "ÉCHEC"
            print(f"{m_name}: {label}")
        else:
            print(f"{m_name}: Non trouvé")

if __name__ == "__main__":
    test_prediction()
