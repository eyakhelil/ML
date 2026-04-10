import os
import sys
import pandas as pd

# Ajouter le dossier src au path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from backend.src.data_loader import load_data
from backend.src.preprocessing import preprocess
from backend.src.train import run_all_experiments

def rebuild():
    # Changer le dossier de travail vers backend
    os.chdir('backend')
    
    print("--- 1. Chargement des données ---")
    df = load_data("data/raw/student_performance_clean.csv")
    print(f"Colonnes trouvées : {df.columns.tolist()}")
    
    print("--- 2. Prétraitement et Feature Engineering ---")
    preprocess(df, save_dir="data/processed")
    
    print("--- 3. Entraînement des modèles ---")
    run_all_experiments()
    print("--- Terminé ! ---")

if __name__ == "__main__":
    rebuild()
