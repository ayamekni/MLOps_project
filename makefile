VENV_DIR = venv
PYTHON = $(VENV_DIR)/Scripts/python.exe
PIP = $(VENV_DIR)/Scripts/pip.exe
FASTAPI_APP = app:app

.PHONY: venv
venv:
	python -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip

.PHONY: install
install: venv
	$(PIP) install -r requirements.txt
	$(PIP) install "uvicorn[standard]" fastapi scikit-learn joblib pandas

.PHONY: run
run: venv
	$(PYTHON) -m uvicorn $(FASTAPI_APP) --reload

.PHONY: clean
clean:
	rm -rf __pycache__ $(VENV_DIR)

.PHONY: test-train
test-train:
	curl -X POST http://127.0.0.1:8000/train

.PHONY: test-predict
test-predict:
	curl -X POST http://127.0.0.1:8000/predict \
	-H "Content-Type: application/json" \
	-d '{"features": [{"GENDER": 1, "AGE": 50, "SMOKING": 1}]}'

.PHONY: test-load
test-load:
	curl -X GET http://127.0.0.1:8000/load_model
