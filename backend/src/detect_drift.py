import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import os

mlflow.set_tracking_uri("./mlruns")
mlflow.set_experiment("monitoring_drift")

print("Chargement des données...", flush=True)
df = pd.read_csv("data/raw/student_performance_clean.csv")
df["pass"] = (df["G3"] >= 10).astype(int)

features_all = [f for f in [
    "absences","studytime","failures","goout","health",
    "famrel","freetime","G1","G2","traveltime",
    "school","sex","address","famsize","reason","guardian",
    "schoolsup","famsup","paid","activities","nursery",
    "higher","internet","romantic"
] if f in df.columns]

X = df[features_all].copy()
y = df["pass"]
for col in X.select_dtypes(include=["object"]).columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Features numériques disponibles — défini APRÈS X_train
features_num = [f for f in [
    "absences","studytime","failures","goout","health",
    "famrel","freetime","G1","G2","traveltime"
] if f in X_train.columns]

print(f"Features numériques : {features_num}", flush=True)
print(f"Train: {X_train.shape}, Test: {X_test.shape}", flush=True)

os.makedirs("reports", exist_ok=True)

# ── Simuler le drift ──────────────────────────────────────────────────────────
print("\nSimulation du drift...", flush=True)
X_prod = X_test.copy()
for col in features_num[:3]:
    X_prod[col] = X_prod[col] * 1.6 + np.random.normal(0, 0.5, len(X_prod))

print(f"Moyenne absences - Ref: {X_train['absences'].mean():.3f} | Prod: {X_prod['absences'].mean():.3f}", flush=True)
print(f"Moyenne G1       - Ref: {X_train['G1'].mean():.3f}      | Prod: {X_prod['G1'].mean():.3f}", flush=True)

drift_share = 0.0
n_drifted   = 0
n_total     = len(features_num)

# ── Partie 1 — Evidently ─────────────────────────────────────────────────────
print("\nGénération du rapport Evidently...", flush=True)
try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    from evidently.metrics import DatasetDriftMetric

    with mlflow.start_run(run_name="drift_check_evidently"):
        report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
        report.run(
            reference_data=X_train[features_num],
            current_data=X_prod[features_num]
        )
        report.save_html("reports/drift_report.html")
        mlflow.log_artifact("reports/drift_report.html")
        print("Rapport HTML sauvegardé ✅", flush=True)

        score_report = Report(metrics=[DatasetDriftMetric()])
        score_report.run(
            reference_data=X_train[features_num],
            current_data=X_prod[features_num]
        )
        result        = score_report.as_dict()
        drift_share   = result["metrics"][0]["result"]["drift_share"]
        dataset_drift = result["metrics"][0]["result"]["dataset_drift"]
        n_drifted     = result["metrics"][0]["result"]["number_of_drifted_columns"]
        n_total       = result["metrics"][0]["result"]["number_of_columns"]

        mlflow.log_metric("drift_share",     drift_share)
        mlflow.log_metric("drifted_columns", n_drifted)
        mlflow.log_metric("total_columns",   n_total)
        mlflow.log_metric("dataset_drifted", int(dataset_drift))

        print(f"Drift share    : {drift_share:.2%}", flush=True)
        print(f"Colonnes driftées : {n_drifted}/{n_total}", flush=True)
        print(f"Dataset drifted   : {dataset_drift}", flush=True)

except Exception as e:
    print(f"Evidently erreur : {e}", flush=True)
    print("Utilisation KS-test uniquement", flush=True)

# ── Partie 2 — KS-test par feature ───────────────────────────────────────────
print("\nKS-test par feature...", flush=True)

ks_results = []
with mlflow.start_run(run_name="drift_check_kstest"):
    for col in features_num:
        stat, pvalue = stats.ks_2samp(X_train[col], X_prod[col])
        drifted = bool(pvalue < 0.05)
        ks_results.append({
            "feature": col,
            "ks_stat": round(stat, 4),
            "p_value": round(pvalue, 4),
            "drifted": drifted
        })
        mlflow.log_metric(f"ks_pvalue_{col}", round(pvalue, 4))
        mlflow.log_metric(f"ks_stat_{col}",   round(stat, 4))

    df_ks = pd.DataFrame(ks_results)
    df_ks.to_csv("reports/ks_drift_results.csv", index=False)
    mlflow.log_artifact("reports/ks_drift_results.csv")

    n_drifted_ks = int(df_ks["drifted"].sum())
    drift_share_ks = n_drifted_ks / len(features_num)
    mlflow.log_metric("ks_drifted_features", n_drifted_ks)
    mlflow.log_metric("ks_drift_share",      round(drift_share_ks, 4))

    print(f"\n{'Feature':>12} | {'KS Stat':>8} | {'P-value':>8} | Drifted", flush=True)
    print("-"*50, flush=True)
    for _, row in df_ks.iterrows():
        status = "⚠️  OUI" if row["drifted"] else "✅  NON"
        print(f"{row['feature']:>12} | {row['ks_stat']:>8.4f} | {row['p_value']:>8.4f} | {status}", flush=True)

    print(f"\nFeatures driftées (KS) : {n_drifted_ks}/{len(features_num)} ({drift_share_ks:.0%})", flush=True)

# Utiliser drift_share KS si Evidently n'a pas fonctionné
if drift_share == 0.0:
    drift_share = drift_share_ks

# ── Partie 3 — Décision ──────────────────────────────────────────────────────
print("\n" + "="*50, flush=True)
print("DÉCISION DE RÉ-ENTRAÎNEMENT", flush=True)
print("="*50, flush=True)

SEUIL_DRIFT   = 0.30
SEUIL_WARNING = 0.15

with mlflow.start_run(run_name="drift_decision"):
    mlflow.log_metric("drift_share",   round(drift_share, 4))
    mlflow.log_metric("seuil_drift",   SEUIL_DRIFT)
    mlflow.log_metric("seuil_warning", SEUIL_WARNING)

    if drift_share > SEUIL_DRIFT:
        print(f"🔴 CRITIQUE : drift={drift_share:.2%} > seuil={SEUIL_DRIFT:.0%}", flush=True)
        print("→ Ré-entraînement DÉCLENCHÉ !", flush=True)
        mlflow.log_metric("retrain_triggered", 1)
        mlflow.log_metric("alert_level", 2)
    elif drift_share > SEUIL_WARNING:
        print(f"🟡 ATTENTION : drift={drift_share:.2%} > warning={SEUIL_WARNING:.0%}", flush=True)
        print("→ Surveillance renforcée", flush=True)
        mlflow.log_metric("retrain_triggered", 0)
        mlflow.log_metric("alert_level", 1)
    else:
        print(f"🟢 OK : drift={drift_share:.2%} — modèle stable", flush=True)
        mlflow.log_metric("retrain_triggered", 0)
        mlflow.log_metric("alert_level", 0)

print("\n✅ Détection de drift terminée !", flush=True)
print("CSV KS-test : reports/ks_drift_results.csv", flush=True)
print("MLflow UI   : http://localhost:5000 → monitoring_drift", flush=True)