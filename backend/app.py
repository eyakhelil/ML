import os
import pandas as pd
import mlflow
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi import File, UploadFile
import io

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
    df = load_data("data/raw/student_performance_clean.csv")
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
@app.get("/data/preview")
def data_preview():
    try:
        df = pd.read_csv("data/raw/student_performance_clean.csv")
        return {
            "columns": df.columns.tolist(),
            "rows": df.head(10).to_dict(orient="records"),
            "total": len(df),
            "shape": {"rows": df.shape[0], "cols": df.shape[1]}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/data/upload")
async def upload_data(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        
        # Détecter séparateur si nécessaire
        if df.shape[1] == 1:
            df = pd.read_csv(io.StringIO(contents.decode("utf-8")), sep=";")
        
        result = {
            "total":        len(df),
            "cols":         len(df.columns),
            "columns":      df.columns.tolist(),
            "missing":      int(df.isnull().sum().sum()),
            "success_rate": None
        }
        
        if "G3" in df.columns:
            df["pass"] = (df["G3"] >= 10).astype(int)
            result["success_rate"] = round(float(df["pass"].mean() * 100), 1)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/data/stats")
def data_stats():
    try:
        # Essayer les deux séparateurs
        try:
            df = pd.read_csv("data/raw/student_performance_clean.csv", sep=";")
            if "G3" not in df.columns:
                df = pd.read_csv("data/raw/student_performance_clean.csv", sep=",")
        except:
            df = pd.read_csv("data/raw/student_performance_clean.csv", sep=",")
        
        print("Colonnes:", df.columns.tolist())
        print("Forme:", df.shape)
        
        if "G3" not in df.columns:
            raise HTTPException(status_code=500, 
                detail=f"Colonne G3 introuvable. Colonnes disponibles: {df.columns.tolist()}")
        
        df["pass"] = (df["G3"] >= 10).astype(int)
        passed = int(df["pass"].sum())
        failed = int((df["pass"] == 0).sum())
        
        return {
            "total_students":  int(len(df)),
            "success_rate":    round(float(df["pass"].mean() * 100), 1),
            "avg_g1":          round(float(df["G1"].mean()), 2),
            "avg_g2":          round(float(df["G2"].mean()), 2),
            "avg_g3":          round(float(df["G3"].mean()), 2),
            "avg_absences":    round(float(df["absences"].mean()), 2),
            "max_absences":    int(df["absences"].max()),
            "passed":          passed,
            "failed":          failed,
            "g3_distribution": {str(int(k)): int(v) for k, v in df["G3"].value_counts().sort_index().items()},
            "absences_bins": {
                "0-5":   int(((df["absences"] >= 0)  & (df["absences"] <= 5)).sum()),
                "6-10":  int(((df["absences"] >= 6)  & (df["absences"] <= 10)).sum()),
                "11-20": int(((df["absences"] >= 11) & (df["absences"] <= 20)).sum()),
                "21+":   int((df["absences"] > 20).sum()),
            },
            "studytime_dist":  {str(int(k)): int(v) for k, v in df["studytime"].value_counts().sort_index().items()},
            "school_dist":     {str(k): int(v) for k, v in df["school"].value_counts().items()},
            "sex_dist":        {str(k): int(v) for k, v in df["sex"].value_counts().items()},
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)