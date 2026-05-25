import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, ConfusionMatrixDisplay
)
import os

# ── Configuration MLflow ──────────────────────────────────────────────────────
mlflow.set_tracking_uri("./mlruns")
mlflow.set_experiment("student_performance_mlops")

# ── Chargement et préparation ─────────────────────────────────────────────────
print("Chargement des données...", flush=True)
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
print(f"Train: {X_train.shape}, Test: {X_test.shape}", flush=True)

os.makedirs("reports", exist_ok=True)

# ── Fonction d'entraînement avec logging complet ──────────────────────────────
def train_and_log(model, params, run_name):
    print(f"\nEntraînement : {run_name}", flush=True)
    
    with mlflow.start_run(run_name=run_name):
        
        # 1. Logger les paramètres
        mlflow.log_params(params)
        
        # 2. Entraînement
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        # 3. Logger les métriques
        metrics = {
            "accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "f1_score":  round(f1_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "recall":    round(recall_score(y_test, y_pred), 4),
        }
        if y_proba is not None:
            metrics["roc_auc"] = round(roc_auc_score(y_test, y_proba), 4)
        
        mlflow.log_metrics(metrics)
        
        # 4. Logger la matrice de confusion comme artefact
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred,
            display_labels=["Échec", "Succès"],
            ax=ax, colorbar=False, cmap="Blues"
        )
        ax.set_title(f"Matrice de confusion — {run_name}")
        plt.tight_layout()
        cm_path = f"reports/confusion_matrix_{run_name}.png"
        plt.savefig(cm_path, dpi=120, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(cm_path)
        
        # 5. Logger le rapport de classification
        report = classification_report(
            y_test, y_pred,
            target_names=["Échec", "Succès"]
        )
        report_path = f"reports/classification_report_{run_name}.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"Run : {run_name}\n")
            f.write("="*50 + "\n")
            f.write(report)
        mlflow.log_artifact(report_path)
        
        # 6. Logger le modèle
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="student_performance_model"
        )
        
        print(f"  Accuracy={metrics['accuracy']} | F1={metrics['f1_score']} | ROC-AUC={metrics.get('roc_auc','N/A')}", flush=True)
        return metrics, mlflow.active_run().info.run_id

# ── Partie 2 — 4 runs avec configurations différentes ────────────────────────
configs = [
    {
        "model": RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42),
        "params": {"model_type":"RandomForest","n_estimators":50,"max_depth":3,"test_size":0.2,"random_state":42},
        "run_name": "rf_50trees_depth3"
    },
    {
        "model": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
        "params": {"model_type":"RandomForest","n_estimators":200,"max_depth":10,"test_size":0.2,"random_state":42},
        "run_name": "rf_200trees_depth10"
    },
    {
        "model": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
        "params": {"model_type":"GradientBoosting","n_estimators":100,"learning_rate":0.1,"test_size":0.2,"random_state":42},
        "run_name": "gb_100trees_lr01"
    },
    {
        "model": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        "params": {"model_type":"LogisticRegression","C":1.0,"max_iter":1000,"test_size":0.2,"random_state":42},
        "run_name": "lr_C1_baseline"
    },
]

results = []
run_ids = []

for cfg in configs:
    metrics, run_id = train_and_log(cfg["model"], cfg["params"], cfg["run_name"])
    results.append({"run_name": cfg["run_name"], "run_id": run_id, **metrics})
    run_ids.append(run_id)

# ── Afficher le tableau des résultats ────────────────────────────────────────
print("\n" + "="*70, flush=True)
print("RÉSUMÉ DES RUNS", flush=True)
print("="*70, flush=True)
df_results = pd.DataFrame(results)
print(df_results.to_string(index=False), flush=True)

# ── Identifier le meilleur run ───────────────────────────────────────────────
from mlflow.tracking import MlflowClient

client = MlflowClient()
experiment = client.get_experiment_by_name("student_performance_mlops")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.accuracy DESC"],
    max_results=5
)
best_run = runs[0]
print(f"\nMeilleur run : {best_run.info.run_name}", flush=True)
print(f"Accuracy     : {best_run.data.metrics['accuracy']:.4f}", flush=True)
print(f"Run ID       : {best_run.info.run_id}", flush=True)

# Sauvegarder le best run ID pour la suite
with open("reports/best_run_id.txt", "w") as f:
    f.write(best_run.info.run_id)

print("\n✅ Partie 1 & 2 terminées — résultats dans MLflow", flush=True)
print("Ouvrez http://localhost:5000 pour visualiser", flush=True)