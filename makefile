# Variables
VENV_DIR = venv
PYTHON = $(VENV_DIR)/Scripts/python.exe
PIP = $(VENV_DIR)/Scripts/pip.exe
FLASK_APP = app.py
FLASK_ENV = development

# Create the virtual environment
.PHONY: venv
venv:
	python -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip

# Install dependencies
.PHONY: install
install: venv
	$(PIP) install -r requirements.txt

# Run Flask server
.PHONY: run
run: venv
	FLASK_APP=$(FLASK_APP) FLASK_ENV=$(FLASK_ENV) $(PYTHON) -m flask run

# Clean temporary files
.PHONY: clean
clean:
	rm -rf __pycache__ $(VENV_DIR)

# Test the /train endpoint
.PHONY: test-train
test-train:
	curl -X POST http://127.0.0.1:5000/train

# Test the /predict endpoint
.PHONY: test-predict
test-predict:
	curl -X POST http://127.0.0.1:5000/predict \
	-H "Content-Type: application/json" \
	-d '{"features": [{"GENDER": 1, "AGE": 50, "SMOKING": 1}]}'

# Test the /load_model endpoint
.PHONY: test-load
test-load:
	curl -X GET http://127.0.0.1:5000/load_model