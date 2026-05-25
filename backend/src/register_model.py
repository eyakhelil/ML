import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("./mlruns")
client = MlflowClient()

# Lire le meilleur run ID
with open("reports/best_run_id.txt", "r") as f:
    best_run_id = f.read().strip()

print(f"Meilleur run ID : {best_run_id}", flush=True)

# ── Étape 1 — Enregistrer le modèle ──────────────────────────────────────────
model_uri = f"runs:/{best_run_id}/model"
registered = mlflow.register_model(
    model_uri=model_uri,
    name="student_performance_model"
)
print(f"Version enregistrée : {registered.version}", flush=True)

# ── Étape 2 — Ajouter description et tags ────────────────────────────────────
client.update_registered_model(
    name="student_performance_model",
    description="Modèle de classification — Prédiction réussite/échec étudiants"
)
client.set_model_version_tag(
    name="student_performance_model",
    version=registered.version,
    key="validated_by",
    value="equipe_data"
)
client.set_model_version_tag(
    name="student_performance_model",
    version=registered.version,
    key="dataset",
    value="UCI_Student_Performance"
)
print("Tags ajoutés ✅", flush=True)

# ── Étape 3 — Promouvoir en Staging ──────────────────────────────────────────
client.transition_model_version_stage(
    name="student_performance_model",
    version=registered.version,
    stage="Staging",
    archive_existing_versions=False
)
print(f"Modèle v{registered.version} promu en STAGING ✅", flush=True)

# ── Étape 4 — Validation et promotion en Production ──────────────────────────
SEUIL_PRODUCTION = 0.85

experiment = client.get_experiment_by_name("student_performance_mlops")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.accuracy DESC"],
    max_results=1
)
best_run = runs[0]
acc = best_run.data.metrics["accuracy"]

print(f"\nValidation — Accuracy: {acc:.4f} | Seuil: {SEUIL_PRODUCTION}", flush=True)

if acc >= SEUIL_PRODUCTION:
    client.transition_model_version_stage(
        name="student_performance_model",
        version=registered.version,
        stage="Production",
        archive_existing_versions=True
    )
    print(f"✅ Modèle v{registered.version} promu en PRODUCTION !", flush=True)
    print(f"   Accuracy={acc:.4f} >= seuil={SEUIL_PRODUCTION}", flush=True)
else:
    print(f"❌ Modèle non promu : accuracy={acc:.3f} < seuil={SEUIL_PRODUCTION}", flush=True)

print("\n✅ Model Registry terminé !", flush=True)
print("Vérifiez dans MLflow UI → Models", flush=True)