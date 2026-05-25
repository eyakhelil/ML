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
    client = mlflow.tracking.MlflowClient()
    all_runs = []
    for exp_name in ["student_performance_mlops", "student_performance_classification"]:
        exp = client.get_experiment_by_name(exp_name)
        if not exp:
            continue
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["metrics.accuracy DESC"]
        )
        for r in runs[:10]:
            all_runs.append({
                "run_id":   r.info.run_id,
                "run_name": r.info.run_name,
                "params":   r.data.params,
                "metrics":  r.data.metrics,
                "experiment": exp_name
            })
    return all_runs
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
@app.get("/task4/results")
def task4_results():
    """Retourne les résultats de la Tâche 4 — Random Forest Analysis."""
    import json
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    import numpy as np

    df = pd.read_csv("data/raw/student_performance_clean.csv")
    df["pass"] = (df["G3"] >= 10).astype(int)

    features = [f for f in [
        "absences","studytime","failures","goout","Dalc","Walc",
        "health","famrel","freetime","G1","G2","school","sex",
        "address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob",
        "reason","guardian","traveltime","schoolsup","famsup","paid",
        "activities","nursery","higher","internet","romantic"
    ] if f in df.columns]

    X = df[features].copy()
    y = df["pass"]
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Q1 — Feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    imp = dict(zip(features, rf.feature_importances_.tolist()))
    top3 = sorted(imp, key=imp.get, reverse=True)[:3]

    # Q2 — Stabilité
    seeds = [0,1,42,123,456,789,1000,2024,9999,31415]
    stability_details = []
    accs, f1s = [], []
    for seed in seeds:
        rf_s = RandomForestClassifier(n_estimators=100, random_state=seed)
        rf_s.fit(X_train, y_train)
        acc = accuracy_score(y_test, rf_s.predict(X_test))
        f1  = f1_score(y_test, rf_s.predict(X_test))
        accs.append(acc); f1s.append(f1)
        stability_details.append({"seed":seed,"accuracy":round(acc,4),"f1":round(f1,4)})

    # Q3 — Erreurs
    y_pred  = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:,1]
    y_test_arr = y_test.values
    X_test_arr = X_test.values
    mask = y_pred != y_test_arr
    examples = []
    for idx in np.where(mask)[0][:3]:
        row = dict(zip(features, X_test_arr[idx]))
        examples.append({
            "G1": int(row.get("G1",0)),
            "G2": int(row.get("G2",0)),
            "absences": int(row.get("absences",0)),
            "failures": int(row.get("failures",0)),
            "true_label": int(y_test_arr[idx]),
            "pred_label": int(y_pred[idx]),
            "proba": round(float(y_proba[idx]),3)
        })

    # Q4 — Biais/Variance
    configs = [(10,2),(10,5),(50,5),(50,None),(100,3),(100,5),(100,10),(100,None),(200,5),(200,None)]
    bv_results = []
    for n_est, depth in configs:
        rf_c = RandomForestClassifier(n_estimators=n_est, max_depth=depth, random_state=42)
        rf_c.fit(X_train, y_train)
        tr = accuracy_score(y_train, rf_c.predict(X_train))
        te = accuracy_score(y_test,  rf_c.predict(X_test))
        b  = round(1-tr, 4)
        v  = round(tr-te, 4)
        d  = "Underfitting" if b>0.05 else "Overfitting" if v>0.05 else "Equilibré ✅"
        bv_results.append({
            "n_estimators": n_est,
            "max_depth": str(depth) if depth else "None",
            "train_acc": round(tr,4),
            "test_acc": round(te,4),
            "biais": b,
            "variance": v,
            "diagnostic": d
        })

    # Q5 — RF vs DT
    rf_f = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_f.fit(X_train, y_train); rf_p = rf_f.predict(X_test)
    dt = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt.fit(X_train, y_train);   dt_p = dt.predict(X_test)

    comparison = [
        {"metric":"Accuracy",  "rf":round(accuracy_score(y_test,rf_p),4),  "dt":round(accuracy_score(y_test,dt_p),4)},
        {"metric":"F1-score",  "rf":round(f1_score(y_test,rf_p),4),        "dt":round(f1_score(y_test,dt_p),4)},
        {"metric":"Precision", "rf":round(precision_score(y_test,rf_p),4), "dt":round(precision_score(y_test,dt_p),4)},
        {"metric":"Recall",    "rf":round(recall_score(y_test,rf_p),4),    "dt":round(recall_score(y_test,dt_p),4)},
    ]

    return {
        "top3": top3,
        "importances": {k: round(v,4) for k,v in sorted(imp.items(), key=lambda x: x[1], reverse=True)},
        "stability": {
            "accuracy_mean": round(np.mean(accs),4),
            "accuracy_std":  round(np.std(accs),6),
            "f1_mean":       round(np.mean(f1s),4),
            "f1_std":        round(np.std(f1s),6),
            "details":       stability_details
        },
        "errors": {
            "total": int(mask.sum()),
            "fp":    int(((y_test_arr==0) & mask).sum()),
            "fn":    int(((y_test_arr==1) & mask).sum()),
            "examples": examples
        },
        "bias_variance": bv_results,
        "comparison": comparison
    }
@app.get("/drift/results")
def drift_results():
    """Retourne les résultats de détection de drift."""
    try:
        df_ks = pd.read_csv("reports/ks_drift_results.csv")
        ks_list = df_ks.to_dict(orient="records")
        n_drifted  = int(df_ks["drifted"].sum())
        drift_share = n_drifted / len(df_ks)
        
        if drift_share > 0.3:
            status = "CRITIQUE"
        elif drift_share > 0.15:
            status = "ATTENTION"
        else:
            status = "OK"
        
        return {
            "total_features":   len(df_ks),
            "drifted_features": n_drifted,
            "drift_share":      round(drift_share, 4),
            "status":           status,
            "ks_results":       ks_list,
            "retrain_needed":   drift_share > 0.3
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Lancez d'abord detect_drift.py")

@app.post("/drift/run")
def run_drift():
    """Lance la détection de drift."""
    import subprocess
    subprocess.Popen(["python", "src/detect_drift.py"])
    return {"status": "drift detection started"}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)