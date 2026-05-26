Write-Host "Pipeline MLOps" -ForegroundColor Cyan
Write-Host "1. Entrainement..." -ForegroundColor Yellow
python src/train_mlops.py
Write-Host "2. Register..." -ForegroundColor Yellow
python src/register_model.py
Write-Host "3. Drift..." -ForegroundColor Yellow
python src/detect_drift.py
Write-Host "Pipeline termine!" -ForegroundColor Green