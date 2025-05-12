# Configuration de l'environnement virtuel
VENV_DIR = venv
PYTHON = $(VENV_DIR)/Scripts/python.exe
PIP = $(VENV_DIR)/Scripts/pip.exe
FASTAPI_APP = app:app

# Création de l'environnement virtuel
.PHONY: venv
venv:
	python -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip

# Installation des dépendances
.PHONY: install
install: venv
	$(PIP) install -r requirements.txt
	$(PIP) install "uvicorn[standard]" fastapi scikit-learn joblib pandas mlflow

# Lancement du serveur FastAPI
.PHONY: run
run:
	$(PYTHON) -m uvicorn $(FASTAPI_APP) --host 127.0.0.1 --port 8000 --reload

# Lancement du serveur MLflow
.PHONY: mlflow-ui
mlflow-ui:
	start /B mlflow ui --host 127.0.0.1 --port 5000

# Test de la prédiction avec cURL
.PHONY: test-predict
test-predict:
	curl -X POST http://127.0.0.1:8000/predict \
	-H "Content-Type: application/json" \
	-d '{"features": [{"GENDER": 1, "AGE": 50, "SMOKING": 1, "YELLOW_FINGERS": 1, "ANXIETY": 1, "PEER_PRESSURE": 1, "CHRONIC_DISEASE": 1, "FATIGUE": 1, "ALLERGY": 1, "WHEEZING": 1, "ALCOHOL_CONSUMING": 1, "COUGHING": 1, "SHORTNESS_OF_BREATH": 1, "SWALLOWING_DIFFICULTY": 1, "CHEST_PAIN": 1}]}'

# Test de réentraînement avec cURL
.PHONY: test-retrain
test-retrain:
	curl -X POST http://127.0.0.1:8000/retrain \
	-H "Content-Type: application/json" \
	-d '{"test_size": 0.2, "C": 1.0, "kernel": "rbf"}'

# Nettoyage des fichiers temporaires et de l'environnement virtuel
.PHONY: clean
clean:
	rmdir /S /Q $(VENV_DIR) __pycache__ 
	del /Q models\svm_model.pkl

# Vérification des logs dans MLflow
.PHONY: mlflow-logs
mlflow-logs:
	start http://127.0.0.1:5000
