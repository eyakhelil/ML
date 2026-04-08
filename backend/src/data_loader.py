import pandas as pd
import os

def load_data(path: str = "data/raw/student-mat.csv") -> pd.DataFrame:
    """Charge le dataset Student Performance."""
    df = pd.read_csv(path, sep=";")
    print(f"Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df

def get_target_distribution(df: pd.DataFrame) -> dict:
    """Retourne la distribution des classes cibles."""
    # On convertit G3 en classification : 0=échec (<10), 1=succès (>=10)
    df["pass"] = (df["G3"] >= 10).astype(int)
    return df["pass"].value_counts().to_dict()

if __name__ == "__main__":
    df = load_data()
    print(df.head())
    print(get_target_distribution(df))