# 🎓 Prédiction des Performances des Étudiants (MLOps)

Ce projet est une application full-stack (FastAPI + React) permettant de prédire la réussite scolaire des étudiants en utilisant plusieurs algorithmes de Machine Learning, tout en assurant un suivi complet des expériences via **MLflow**.

## 🚀 Fonctionnalités Clés

*   **Comparaison Multi-Modèles** : KNN, SVM, Random Forest et Régression Logistique.
*   **Suivi MLOps (MLflow)** : Enregistrement automatique des hyperparamètres, des métriques (Accuracy, F1, ROC-AUC) et des modèles.
*   **Règle Métier Intelligente** : Correction logique automatique. Si l'étudiant a une moyenne $(G1 + G2) / 2 < 10$, le système force la prédiction "Échec" pour garantir la fiabilité académique.
*   **Dashboard de Comparaison** : Visualisation dynamique des performances avec identification automatique du meilleur modèle.

## 🛠️ Architecture Technique

*   **Frontend** : React 19, Recharts (visualisation), Axios.
*   **Backend** : Python 3.x, FastAPI, Scikit-Learn.
*   **Tracking** : MLflow pour l'historique des expériences.
*   **Data** : Utilisation du dataset *Student Performance* (UCI), nettoyé et prétraité (634 lignes).

## 📥 Installation et Démarrage

### 1. Backend (API & ML)
```bash
cd backend
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt

# Lancer le serveur
python app.py
```
*L'API est accessible sur [http://localhost:8000](http://localhost:8000)*

### 2. Frontend (Interface)
```bash
cd frontend
npm install
npm start
```
*L'interface est accessible sur [http://localhost:3000](http://localhost:3000)*

### 3. Visualisation MLflow
```bash
cd backend
mlflow ui
```
*Accédez au tracking sur [http://localhost:5000](http://localhost:5000)*

## 🔄 Automatisation du Ré-entraînement

Pour ré-entraîner tous les modèles sur le dernier dataset (`data/raw/student_performance_clean.csv`) :
```bash
python rebuild_models.py
```

## 📊 Critères de Prédiction
L'IA se base sur 31 variables, dont les plus influentes :
*   **Académique** : Notes G1/G2, échecs passés.
*   **Socio-économique** : Éducation des parents, temps de trajet.
*   **Style de vie** : Consommation d'alcool, temps d'étude, sorties.

---
*Projet développé dans le cadre de l'implémentation d'algorithmes de classification avec suivi MLflow.*
