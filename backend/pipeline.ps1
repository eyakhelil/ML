Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   Pipeline MLOps — Student Performance  " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "`n[1/3] Entraînement des modèles..." -ForegroundColor Yellow
python src/train_mlops.py
if ($LASTEXITCODE -ne 0) { Write-Host "ERREUR entraînement" -ForegroundColor Red; exit 1 }
Write-Host "Entraînement OK ✅" -ForegroundColor Green

Write-Host "`n[2/3] Enregistrement du meilleur modèle..." -ForegroundColor Yellow
python src/register_model.py
if ($LASTEXITCODE -ne 0) { Write-Host "ERREUR register" -ForegroundColor Red; exit 1 }
Write-Host "Register OK ✅" -ForegroundColor Green

Write-Host "`n[3/3] Détection de drift..." -ForegroundColor Yellow
python src/detect_drift.py
if ($LASTEXITCODE -ne 0) { Write-Host "ERREUR drift" -ForegroundColor Red; exit 1 }
Write-Host "Drift OK ✅" -ForegroundColor Green

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "   Pipeline MLOps terminé avec succès !  " -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan