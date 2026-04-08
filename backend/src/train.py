import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix
)
from sklearn.decomposition import PCA
import joblib
import os
import json

# Chemin MLflow
mlflow.set_tracking_uri("../mlruns")
mlflow.set_experiment("student_performance_classification")

def load_processed_data(data_dir: str = "../data/processed"):
    X_train = pd.read_csv(f"{data_dir}/X_train.csv").values
    X_test  = pd.read_csv(f"{data_dir}/X_test.csv").values
    y_train = pd.read_csv(f"{data_dir}/y_train.csv").values.ravel()
    y_test  = pd.read_csv(f"{data_dir}/y_test.csv").values.ravel()
    return X_train, X_test, y_train, y_test

def compute_metrics(y_true, y_pred, y_proba=None):
    metrics = {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "f1_score":  round(f1_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall":    round(recall_score(y_true, y_pred), 4),
    }
    if y_proba is not None:
        metrics["roc_auc"] = round(roc_auc_score(y_true, y_proba), 4)
    cm = confusion_matrix(y_true, y_pred).tolist()
    return metrics, cm

def train_and_log(model, params, model_name, X_train, X_test, y_train, y_test,
                  use_pca=False, pca_components=10):
    
    # Configurer MLflow ici, pas au niveau global
    mlflow.set_tracking_uri("../mlruns")
    mlflow.set_experiment("student_performance_classification")
    
    with mlflow.start_run(run_name=model_name):
        # Log des paramètres
        mlflow.log_params(params)
        mlflow.log_param("use_pca", use_pca)
        mlflow.log_param("model_type", model_name)
        
        Xtr, Xte = X_train.copy(), X_test.copy()
        
        # Réduction de dimension optionnelle
        if use_pca:
            mlflow.log_param("pca_components", pca_components)
            pca = PCA(n_components=pca_components, random_state=42)
            Xtr = pca.fit_transform(Xtr)
            Xte = pca.transform(Xte)
            variance = round(sum(pca.explained_variance_ratio_), 4)
            mlflow.log_metric("pca_explained_variance", variance)
            joblib.dump(pca, f"../models/{model_name}_pca.pkl")
        
        # Entraînement
        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)
        
        # Probabilités pour ROC-AUC
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(Xte)[:, 1]
        
        # Métriques
        metrics, cm = compute_metrics(y_test, y_pred, y_proba)
        mlflow.log_metrics(metrics)
        
        # Log de la matrice de confusion en JSON
        mlflow.log_dict({"confusion_matrix": cm}, "confusion_matrix.json")
        
        # Sauvegarde du modèle
        os.makedirs("../models", exist_ok=True)
        model_path = f"../models/{model_name}.pkl"
        joblib.dump(model, model_path)
        mlflow.sklearn.log_model(model, artifact_path="model",
                                  registered_model_name=model_name)
        
        print(f"[{model_name}] Accuracy={metrics['accuracy']} | F1={metrics['f1_score']} | ROC-AUC={metrics.get('roc_auc','N/A')}")
        return metrics

def run_all_experiments():
    X_train, X_test, y_train, y_test = load_processed_data()
    
    results = []

    # ── KNN ───────────────────────────────────────────────────────────────────
    for k in [3, 5, 9]:
        params = {"n_neighbors": k, "metric": "euclidean"}
        m = train_and_log(
            KNeighborsClassifier(**params), params,
            f"KNN_k{k}", X_train, X_test, y_train, y_test
        )
        results.append({"model": f"KNN (k={k})", **m})

    # KNN avec PCA
    params = {"n_neighbors": 5, "metric": "euclidean"}
    m = train_and_log(
        KNeighborsClassifier(**params), params,
        "KNN_k5_PCA", X_train, X_test, y_train, y_test,
        use_pca=True, pca_components=10
    )
    results.append({"model": "KNN (k=5, PCA)", **m})

    # ── SVM ───────────────────────────────────────────────────────────────────
    for kernel in ["rbf", "linear", "poly"]:
        for C in [0.1, 1.0, 10.0]:
            params = {"kernel": kernel, "C": C, "probability": True}
            m = train_and_log(
                SVC(**params), params,
                f"SVM_{kernel}_C{C}", X_train, X_test, y_train, y_test
            )
            results.append({"model": f"SVM ({kernel}, C={C})", **m})

    # ── Random Forest ─────────────────────────────────────────────────────────
    for n_trees in [50, 100, 200]:
        for max_depth in [None, 5, 10]:
            depth_label = max_depth if max_depth else "None"
            params = {"n_estimators": n_trees, "max_depth": max_depth, "random_state": 42}
            m = train_and_log(
                RandomForestClassifier(**params), params,
                f"RF_{n_trees}trees_depth{depth_label}",
                X_train, X_test, y_train, y_test
            )
            results.append({"model": f"RF ({n_trees} arbres, depth={depth_label})", **m})

    # ── Logistic Regression ───────────────────────────────────────────────────
    for C in [0.01, 0.1, 1.0, 10.0]:
        for solver in ["lbfgs", "liblinear"]:
            params = {"C": C, "solver": solver, "max_iter": 1000, "random_state": 42}
            m = train_and_log(
                LogisticRegression(**params), params,
                f"LR_C{C}_{solver}", X_train, X_test, y_train, y_test
            )
            results.append({"model": f"LR (C={C}, {solver})", **m})

    # Sauvegarde du tableau comparatif
    df_results = pd.DataFrame(results)
    df_results.to_csv("../data/processed/results_comparison.csv", index=False)
    print("\n── Résumé ──────────────────────────────────────────────────────")
    print(df_results.sort_values("f1_score", ascending=False).to_string(index=False))
    
    return df_results

if __name__ == "__main__":
    run_all_experiments()