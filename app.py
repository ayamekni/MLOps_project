from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from utils.data_loader import load_and_preprocess_data
import os

# Initialize FastAPI app
app = FastAPI()

# Global model variables
model_path = "models/svm_model.pkl"
svm_model = None

# Define the data structure for prediction input
class FeaturesInput(BaseModel):
    features: List[Dict[str, float]]

# Define the data structure for retraining parameters
class RetrainParams(BaseModel):
    test_size: float = 0.2
    C: float = 1.0
    kernel: str = "rbf"

# Set up MLflow tracking
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Lung_Cancer_Prediction")

# Home endpoint
@app.get("/")
def home():
    return {"message": "Welcome to the FastAPI prediction service. Use /docs to test."}

# Prediction endpoint
@app.post("/predict")
def predict(input_data: FeaturesInput):
    global svm_model

    # Check if model is loaded
    if svm_model is None:
        try:
            svm_model = joblib.load(model_path)
            print("✅ Model loaded successfully.")
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail="Model not found. Please train the model first.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

    # Convert input data to DataFrame
    df = pd.DataFrame(input_data.features)

    try:
        # Make predictions
        predictions = svm_model.predict(df)
        probs = svm_model.predict_proba(df)

        return {
            "predictions": predictions.tolist(),
            "probabilities": probs.tolist()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Retrain endpoint
@app.post("/retrain")
def retrain(params: RetrainParams):
    global svm_model

    try:
        # Load and preprocess the data
        df = load_and_preprocess_data()
        X = df.drop("LUNG_CANCER", axis=1)
        y = df["LUNG_CANCER"]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params.test_size, random_state=42)

        # Start MLflow experiment
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("C", params.C)
            mlflow.log_param("kernel", params.kernel)
            mlflow.log_param("test_size", params.test_size)

            # Train the model
            svm_model = SVC(probability=True, kernel=params.kernel, C=params.C)
            svm_model.fit(X_train, y_train)

            # Predictions
            y_pred = svm_model.predict(X_test)

            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            clf_report = classification_report(y_test, y_pred, output_dict=True)
            conf_matrix = confusion_matrix(y_test, y_pred)

            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", clf_report['1']['precision'])
            mlflow.log_metric("recall", clf_report['1']['recall'])
            mlflow.log_metric("f1-score", clf_report['1']['f1-score'])

            # Save model locally
            if not os.path.exists("models"):
                os.makedirs("models")
            joblib.dump(svm_model, model_path)

            # Log the model with MLflow
            mlflow.sklearn.log_model(svm_model, "model", registered_model_name="SVM_LungCancer_Model")

            return {
                "message": "Model retrained successfully.",
                "accuracy": accuracy,
                "confusion_matrix": conf_matrix.tolist(),
                "classification_report": clf_report
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining error: {str(e)}")
