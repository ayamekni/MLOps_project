from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from utils.data_loader import load_and_preprocess_data

app = FastAPI()
model_path = "models/svm_model.pkl"
svm_model = None  # global variable

class FeaturesInput(BaseModel):
    features: List[Dict[str, float]]

class RetrainParams(BaseModel):
    test_size: float = 0.2
    C: float = 1.0
    kernel: str = "rbf"

@app.post("/predict")
def predict(input_data: FeaturesInput):
    global svm_model
    if svm_model is None:
        try:
            svm_model = joblib.load(model_path)
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail="Model not found. Please train the model first.")

    df = pd.DataFrame(input_data.features)
    predictions = svm_model.predict(df)
    probs = svm_model.predict_proba(df)

    return {
        "predictions": predictions.tolist(),
        "probabilities": probs.tolist()
    }

@app.post("/retrain")
def retrain(params: RetrainParams):
    global svm_model
    df = load_and_preprocess_data()
    X = df.drop("LUNG_CANCER", axis=1)
    y = df["LUNG_CANCER"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params.test_size, random_state=42)

    svm_model = SVC(probability=True, kernel=params.kernel, C=params.C)
    svm_model.fit(X_train, y_train)

    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    joblib.dump(svm_model, model_path)

    return {
        "message": "Model retrained successfully.",
        "accuracy": accuracy,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }

@app.get("/")
def home():
    return {"message": "Welcome to the FastAPI prediction service. Use /docs to test."}
