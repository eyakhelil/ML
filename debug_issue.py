import joblib
import pandas as pd
import os

base_dir = r"c:\Users\Administrator\project_studentPerformance\backend"
scaler_path = os.path.join(base_dir, "data", "processed", "scaler.pkl")
encoders_path = os.path.join(base_dir, "data", "processed", "encoders.pkl")

scaler = joblib.load(scaler_path)
encoders = joblib.load(encoders_path)

print("Scaler features:", scaler.feature_names_in_)
print("Encoder keys:", list(encoders.keys()))

# Simuler une prédiction
student = {
    'absences': 4, 'studytime': 2, 'failures': 0, 'goout': 3, 
    'G1': 6, 'G2': 7, 'higher': 'yes', 'internet': 'yes'
}
df = pd.DataFrame([student])

# Normalisation comme dans evaluate.py (AVANT correction)
print("\n--- Avant normalisation des colonnes ---")
print("df columns:", df.columns.tolist())
for col, le in encoders.items():
    if col in df.columns:
        print(f"Encoding {col}")
        df[col] = le.transform(df[col])
    else:
        print(f"Skipping {col} (not in df.columns)")

print("df after encoding attempt:\n", df)

try:
    # Garder seulement les colonnes du scaler
    df_for_scaler = df[scaler.feature_names_in_]
    print("\ndf for scaler:\n", df_for_scaler)
    scaler.transform(df_for_scaler)
    print("\nScaler transform SUCCESS")
except Exception as e:
    print("\nScaler transform FAILED:", e)
