🎓 Prédiction des Performances des Étudiants
📋 Description
Ce projet prédit la performance scolaire des étudiants (réussite ou échec) en fonction de trois facteurs clés :

📅 Assiduité — nombre d'absences
📝 Notes aux examens — notes des 1er et 2ème trimestres (G1, G2)
🙋 Participation en classe — temps de travail et engagement

Le dataset utilisé est le Student Performance Dataset de l'UCI Machine Learning Repository (395 étudiants, 33 variables).

2. Lancer le backend
bashcd backend
python -m venv venv
venv\Scripts\activate         
pip install -r requirements.txt

# Lancer l'API
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
3. Lancer le frontend
bashcd frontend
npm install
npm start
4. Lancer MLflow UI
bashcd backend
mlflow ui --port 5000
Interface Reac thttp://localhost:3000 
API FastAPI (docs) http://localhost:8000/docs
MLflow UI http://localhost:5000

🛠️ Technologies utilisées
Frontend : React
Backend : Python, FastAPI, Uvicorn
MLScikit-learn (KNN, SVM, RF, LR, PCA)
MLOps : MLflow 
Data : Pandas, NumPy
