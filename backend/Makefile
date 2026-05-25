.PHONY: setup train register serve drift test pipeline

setup:
	pip install -r requirements.txt

train:
	python src/train_mlops.py

register:
	python src/register_model.py

serve:
	python src/serve_model.py

drift:
	python src/detect_drift.py

pipeline: train register drift
	@echo "Pipeline MLOps complet execute avec succes"