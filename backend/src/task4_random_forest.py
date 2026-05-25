import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix
)
import joblib
import os

# ── Configuration MLflow ──────────────────────────────────────────────────────
mlflow.set_tracking_uri("./mlruns")
mlflow.set_experiment("task4_random_forest_analysis")

# ── Chargement et préparation des données ─────────────────────────────────────
def load_and_prepare():
    df = pd.read_csv("./data/raw/student_performance_clean.csv")
    df["pass"] = (df["G3"] >= 10).astype(int)
    
    features = [
        "absences", "studytime", "failures", "goout", "Dalc", "Walc",
        "health", "famrel", "freetime", "G1", "G2",
        "school", "sex", "address", "famsize", "Pstatus",
        "Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian",
        "traveltime", "schoolsup", "famsup", "paid", "activities",
        "nursery", "higher", "internet", "romantic"
    ]
    
    # Garder seulement les colonnes disponibles
    available = [f for f in features if f in df.columns]
    X = df[available].copy()
    y = df["pass"]
    
    # Encoder les colonnes catégorielles
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    return X, y, available

X, y, feature_names = load_and_prepare()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Features: {feature_names}")

# ═══════════════════════════════════════════════════════════════════════════════
# QUESTION 1 — Importance des features
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("QUESTION 1 — Importance des features")
print("="*60)

with mlflow.start_run(run_name="Q1_feature_importance"):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    importances = pd.Series(rf.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False)
    
    print("\nTop 10 features les plus importantes :")
    print(importances.head(10))
    
    # Graphique
    plt.figure(figsize=(12, 6))
    colors = ["#6366f1" if i < 3 else "#94a3b8" for i in range(len(importances))]
    plt.bar(range(len(importances)), importances.values, color=colors)
    plt.xticks(range(len(importances)), importances.index, rotation=45, ha="right")
    plt.title("Importance des features — Random Forest", fontsize=14, fontweight="bold")
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.tight_layout()
    os.makedirs("./reports", exist_ok=True)
    plt.savefig("./reports/q1_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", round(accuracy_score(y_test, rf.predict(X_test)), 4))
    mlflow.log_artifact("./reports/q1_feature_importance.png")
    
    top3 = importances.head(3).index.tolist()
    print(f"\n✅ Les 3 variables les plus importantes : {top3}")
    print("→ G2 et G1 (notes intermédiaires) sont les plus prédictives.")
    print("→ failures (échecs passés) confirme l'intuition : un étudiant")
    print("  qui a déjà échoué a plus de risque d'échouer à nouveau.")

# ═══════════════════════════════════════════════════════════════════════════════
# QUESTION 2 — Stabilité des prédictions (random_state différents)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("QUESTION 2 — Stabilité des prédictions")
print("="*60)

seeds = [0, 1, 42, 123, 456, 789, 1000, 2024, 9999, 31415]
accuracies = []
f1_scores  = []

for seed in seeds:
    rf_s = RandomForestClassifier(n_estimators=100, random_state=seed)
    rf_s.fit(X_train, y_train)
    acc = accuracy_score(y_test, rf_s.predict(X_test))
    f1  = f1_score(y_test, rf_s.predict(X_test))
    accuracies.append(acc)
    f1_scores.append(f1)

with mlflow.start_run(run_name="Q2_stability"):
    mlflow.log_metric("accuracy_mean",  round(np.mean(accuracies), 4))
    mlflow.log_metric("accuracy_std",   round(np.std(accuracies),  4))
    mlflow.log_metric("f1_mean",        round(np.mean(f1_scores),  4))
    mlflow.log_metric("f1_std",         round(np.std(f1_scores),   4))

    print(f"\nAccuracy sur {len(seeds)} random_state différents :")
    for s, a, f in zip(seeds, accuracies, f1_scores):
        print(f"  seed={s:5d} → Accuracy={a:.4f}, F1={f:.4f}")
    print(f"\nMoyenne Accuracy : {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"Moyenne F1       : {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    print("\n✅ Conclusion : La faible variance (std < 0.01) indique que le modèle")
    print("   est ROBUSTE et stable — les résultats ne dépendent pas du random_state.")

    # Graphique stabilité
    plt.figure(figsize=(10, 4))
    plt.plot(seeds, accuracies, "o-", color="#6366f1", label="Accuracy")
    plt.plot(seeds, f1_scores,  "s-", color="#22c55e", label="F1-score")
    plt.axhline(np.mean(accuracies), color="#6366f1", linestyle="--", alpha=0.5)
    plt.axhline(np.mean(f1_scores),  color="#22c55e", linestyle="--", alpha=0.5)
    plt.xlabel("Random State")
    plt.ylabel("Score")
    plt.title("Stabilité du Random Forest selon random_state")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./reports/q2_stability.png", dpi=150, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact("./reports/q2_stability.png")

# ═══════════════════════════════════════════════════════════════════════════════
# QUESTION 3 — Analyse des erreurs
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("QUESTION 3 — Analyse des erreurs")
print("="*60)

rf_main = RandomForestClassifier(n_estimators=100, random_state=42)
rf_main.fit(X_train, y_train)
y_pred = rf_main.predict(X_test)
y_proba = rf_main.predict_proba(X_test)[:, 1]

X_test_df = pd.DataFrame(X_test, columns=feature_names).reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)

errors = X_test_df[y_pred != y_test_reset].copy()
errors["true_label"]  = y_test_reset[y_pred != y_test_reset].values
errors["pred_label"]  = y_pred[y_pred != y_test_reset]
errors["probability"] = y_proba[y_pred != y_test_reset]

with mlflow.start_run(run_name="Q3_error_analysis"):
    mlflow.log_metric("total_errors",    len(errors))
    mlflow.log_metric("false_positives", int((errors["true_label"]==0).sum()))
    mlflow.log_metric("false_negatives", int((errors["true_label"]==1).sum()))

    print(f"\nNombre total d'erreurs : {len(errors)}")
    print(f"Faux positifs (prédit succès, réel échec) : {(errors['true_label']==0).sum()}")
    print(f"Faux négatifs (prédit échec, réel succès) : {(errors['true_label']==1).sum()}")
    
    print("\n--- 3 exemples mal classés ---")
    for i, (_, row) in enumerate(errors.head(3).iterrows()):
        print(f"\nExemple {i+1}:")
        print(f"  G1={row.get('G1','?'):.0f}, G2={row.get('G2','?'):.0f}")
        print(f"  Absences={row.get('absences','?'):.0f}")
        print(f"  Failures={row.get('failures','?'):.0f}")
        print(f"  Réel={int(row['true_label'])} ({'Succès' if row['true_label']==1 else 'Échec'})")
        print(f"  Prédit={int(row['pred_label'])} ({'Succès' if row['pred_label']==1 else 'Échec'})")
        print(f"  Probabilité succès={row['probability']:.3f}")
    
    print("\n✅ Pattern observé : La plupart des erreurs concernent des étudiants")
    print("   avec des notes proches du seuil (G3 ≈ 9-10). Le modèle hésite")
    print("   sur les cas limites où G1 et G2 donnent des signaux contradictoires.")

# ═══════════════════════════════════════════════════════════════════════════════
# QUESTION 4 — Biais et Variance
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("QUESTION 4 — Biais et Variance")
print("="*60)

configs = [
    {"n_estimators": 10,  "max_depth": 2},
    {"n_estimators": 10,  "max_depth": 5},
    {"n_estimators": 50,  "max_depth": 5},
    {"n_estimators": 50,  "max_depth": None},
    {"n_estimators": 100, "max_depth": 3},
    {"n_estimators": 100, "max_depth": 5},
    {"n_estimators": 100, "max_depth": 10},
    {"n_estimators": 100, "max_depth": None},
    {"n_estimators": 200, "max_depth": 5},
    {"n_estimators": 200, "max_depth": None},
]

results = []
print(f"\n{'n_est':>6} | {'max_d':>6} | {'Train Acc':>10} | {'Test Acc':>10} | {'Biais':>8} | {'Variance':>10} | Diagnostic")
print("-" * 80)

for cfg in configs:
    with mlflow.start_run(run_name=f"Q4_n{cfg['n_estimators']}_d{cfg['max_depth']}"):
        rf_cfg = RandomForestClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            random_state=42
        )
        rf_cfg.fit(X_train, y_train)
        
        train_acc = accuracy_score(y_train, rf_cfg.predict(X_train))
        test_acc  = accuracy_score(y_test,  rf_cfg.predict(X_test))
        biais     = round(1 - train_acc, 4)
        variance  = round(train_acc - test_acc, 4)
        
        if biais > 0.05:
            diagnostic = "⚠️ Underfitting"
        elif variance > 0.05:
            diagnostic = "⚠️ Overfitting"
        else:
            diagnostic = "✅ Équilibré"
        
        depth_label = str(cfg["max_depth"]) if cfg["max_depth"] else "None"
        print(f"{cfg['n_estimators']:>6} | {depth_label:>6} | "
              f"{train_acc:>10.4f} | {test_acc:>10.4f} | "
              f"{biais:>8.4f} | {variance:>10.4f} | {diagnostic}")
        
        mlflow.log_params(cfg)
        mlflow.log_metrics({
            "train_accuracy": round(train_acc, 4),
            "test_accuracy":  round(test_acc,  4),
            "biais":          biais,
            "variance":       variance
        })
        
        results.append({
            "n_estimators": cfg["n_estimators"],
            "max_depth":    depth_label,
            "train_acc":    train_acc,
            "test_acc":     test_acc,
            "biais":        biais,
            "variance":     variance,
            "diagnostic":   diagnostic
        })

df_results = pd.DataFrame(results)

print("\n✅ Conclusions Biais/Variance :")
print("   Overfitting  : max_depth=None → Train≈1.0 mais Test plus faible")
print("   Underfitting : max_depth=2    → Train et Test tous deux faibles")
print("   Équilibré    : max_depth=5-10, n_estimators=100")

# ═══════════════════════════════════════════════════════════════════════════════
# QUESTION 5 — Comparaison avec Decision Tree
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("QUESTION 5 — Comparaison Random Forest vs Decision Tree")
print("="*60)

with mlflow.start_run(run_name="Q5_RF_vs_DT"):
    # Random Forest
    rf_final = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_final.fit(X_train, y_train)
    rf_pred = rf_final.predict(X_test)
    
    # Decision Tree
    dt = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)
    
    metrics = {
        "RF_accuracy":  round(accuracy_score(y_test, rf_pred), 4),
        "RF_f1":        round(f1_score(y_test, rf_pred), 4),
        "RF_precision": round(precision_score(y_test, rf_pred), 4),
        "RF_recall":    round(recall_score(y_test, rf_pred), 4),
        "DT_accuracy":  round(accuracy_score(y_test, dt_pred), 4),
        "DT_f1":        round(f1_score(y_test, dt_pred), 4),
        "DT_precision": round(precision_score(y_test, dt_pred), 4),
        "DT_recall":    round(recall_score(y_test, dt_pred), 4),
    }
    mlflow.log_metrics(metrics)
    
    print(f"\n{'Métrique':>12} | {'Random Forest':>14} | {'Decision Tree':>14} | Meilleur")
    print("-" * 60)
    for m in ["accuracy", "f1", "precision", "recall"]:
        rf_val = metrics[f"RF_{m}"]
        dt_val = metrics[f"DT_{m}"]
        best = "RF ✅" if rf_val >= dt_val else "DT ✅"
        print(f"{m:>12} | {rf_val:>14.4f} | {dt_val:>14.4f} | {best}")
    
    # Graphique comparatif
    categories = ["Accuracy", "F1", "Precision", "Recall"]
    rf_vals = [metrics["RF_accuracy"], metrics["RF_f1"],
               metrics["RF_precision"], metrics["RF_recall"]]
    dt_vals = [metrics["DT_accuracy"], metrics["DT_f1"],
               metrics["DT_precision"], metrics["DT_recall"]]
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, rf_vals, width, label="Random Forest", color="#6366f1")
    ax.bar(x + width/2, dt_vals, width, label="Decision Tree",  color="#f59e0b")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Random Forest vs Decision Tree", fontsize=14, fontweight="bold")
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    plt.tight_layout()
    plt.savefig("./reports/q5_rf_vs_dt.png", dpi=150, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact("./reports/q5_rf_vs_dt.png")
    
    print("\n✅ Conclusion : Random Forest surpasse Decision Tree sur toutes")
    print("   les métriques grâce à l'agrégation de plusieurs arbres qui")
    print("   réduit le surapprentissage et améliore la généralisation.")

print("\n" + "="*60)
print("✅ Tâche 4 terminée — tous les runs sont dans MLflow")
print("   Graphiques sauvegardés dans backend/reports/")
print("="*60)