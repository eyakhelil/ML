import mlflow.sklearn
import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

print("Chargement du modèle depuis MLflow Registry...", flush=True)

mlflow.set_tracking_uri("./mlruns")

# Charger le modèle en Production depuis le Registry
model = mlflow.sklearn.load_model("models:/student_performance_model/Production")
print("Modèle chargé ✅", flush=True)

app = FastAPI(title="MLflow Model Serving — Student Performance")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FEATURES = [
    "absences","studytime","failures","goout","health","famrel","freetime",
    "G1","G2","school","sex","address","famsize","reason","guardian",
    "traveltime","schoolsup","famsup","paid","activities","nursery",
    "higher","internet","romantic"
]

class PredictRequest(BaseModel):
    dataframe_split: dict

class SimplePredictRequest(BaseModel):
    data: list

@app.get("/ping")
def ping():
    return {"status": "ok", "model": "student_performance_model/Production"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/invocations")
def invocations(req: PredictRequest):
    """Endpoint compatible MLflow serving natif."""
    try:
        columns = req.dataframe_split.get("columns", FEATURES)
        data    = req.dataframe_split.get("data", [])
        df = pd.DataFrame(data, columns=columns)
        preds  = model.predict(df).tolist()
        probas = model.predict_proba(df).tolist() if hasattr(model, "predict_proba") else None
        return {
            "predictions": preds,
            "probabilities": probas,
            "labels": ["Succès" if p == 1 else "Échec" for p in preds]
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict_simple")
def predict_simple(req: SimplePredictRequest):
    """Endpoint simplifié pour tests rapides."""
    try:
        df = pd.DataFrame([req.data], columns=FEATURES)
        pred  = int(model.predict(df)[0])
        proba = float(model.predict_proba(df)[0][1]) if hasattr(model, "predict_proba") else None
        return {
            "prediction":   pred,
            "label":        "Succès ✅" if pred == 1 else "Échec ❌",
            "probability":  round(proba, 4) if proba else None,
            "model":        "student_performance_model/Production"
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    print("Serving sur http://127.0.0.1:1234", flush=True)
    print("Docs : http://127.0.0.1:1234/docs", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=1234)