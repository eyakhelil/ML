import sys
import os
import joblib

# Set paths
base_dir = r"c:\Users\Administrator\project_studentPerformance"
sys.path.append(os.path.join(base_dir, "backend"))

from backend.src.evaluate import predict_student

# Paths to artifacts
model_path = os.path.join(base_dir, "backend", "models", "KNN_k3.pkl")
scaler_path = os.path.join(base_dir, "backend", "data", "processed", "scaler.pkl")
encoders_path = os.path.join(base_dir, "backend", "data", "processed", "encoders.pkl")

# Payload that was causing the error (with "yes" strings and mixed casing)
student_payload = {
    'absences': 4, 'studytime': 2, 'failures': 0, 'goout': 3, 
    'G1': 12, 'G2': 13, 
    'higher': 'yes', 'internet': 'yes', 'Pstatus': 'T'
}

print("--- Testing predict_student with previously problematic payload ---")
try:
    result = predict_student(model_path, scaler_path, encoders_path, student_payload)
    print("SUCCESS! Prediction result:", result)
except Exception as e:
    print("FAILED! Error:", e)
    import traceback
    traceback.print_exc()

# Test with numeric values (as now sent by frontend)
student_payload_numeric = {
    'absences': 4, 'studytime': 2, 'failures': 0, 'goout': 3, 
    'G1': 10, 'G2': 11, 
    'higher': 1, 'internet': 1, 'pstatus': 'T'
}

print("\n--- Testing predict_student with numeric payload ---")
try:
    result = predict_student(model_path, scaler_path, encoders_path, student_payload_numeric)
    print("SUCCESS! Prediction result:", result)
except Exception as e:
    print("FAILED! Error:", e)
