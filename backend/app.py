import os
import pandas as pd
import mlflow
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.preprocessing import preprocess
from src.train import run_all_experiments
from src.evaluate import predict_student
from src.data_loader import load_data

app = FastAPI(title="Student Performance ML API")

# CORS - doit être EN PREMIER
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

mlflow.set_tracking_uri("mlruns")

@app.get("/models")
def list_models():
    try:
        df = pd.read_csv("data/processed/results_comparison.csv").fillna(0)
        return [{
            "model":     str(row.get("model", "")),
            "accuracy":  float(row.get("accuracy", 0)),
            "f1_score":  float(row.get("f1_score", 0)),
            "precision": float(row.get("precision", 0)),
            "recall":    float(row.get("recall", 0)),
            "roc_auc":   float(row.get("roc_auc", 0)),
        } for _, row in df.iterrows()]
    except Exception as e:
        print(f"Erreur list_models: {e}")
        return []

@app.post("/train")
def train_models():
    df = load_data("data/raw/student-mat.csv")
    preprocess(df, save_dir="data/processed")
    results = run_all_experiments()
    return {"status": "ok", "runs": len(results)}

class PredictRequest(BaseModel):
    model_name: str
    student: dict

@app.post("/predict")
def predict(req: PredictRequest):
    print(f"Prédiction demandée pour : {req.model_name}")
    model_path  = f"models/{req.model_name}.pkl"
    scaler_path = "data/processed/scaler.pkl"
    enc_path    = "data/processed/encoders.pkl"
    if not os.path.exists(model_path):
        available = os.listdir("models/")
        raise HTTPException(
            status_code=404,
            detail=f"Modèle '{req.model_name}' introuvable. Disponibles: {available}"
        )
    try:
        result = predict_student(model_path, scaler_path, enc_path, req.student)
        print(f"Résultat : {result}")
        return result
    except Exception as e:
        print(f"Erreur predict: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mlflow/runs")
def get_mlflow_runs():
    try:
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name("student_performance_classification")
        if not exp:
            return []
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["metrics.f1_score DESC"]
        )
        return [{
            "run_id":   r.info.run_id,
            "run_name": r.info.run_name,
            "params":   r.data.params,
            "metrics":  r.data.metrics,
        } for r in runs[:20]]
    except Exception as e:
        print(f"Erreur mlflow: {e}")
        return []

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)