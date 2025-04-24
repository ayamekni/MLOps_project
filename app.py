from flask import Flask, request, jsonify
import joblib
from utils.data_loader import load_and_preprocess_data
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

app = Flask(__name__)

# Placeholder for the trained model
svm_model = None


@app.route('/train', methods=['POST'])
def train_model():
    global svm_model

    # Load and preprocess the dataset
    df = load_and_preprocess_data()
    X = df.drop('LUNG_CANCER', axis=1)
    y = df['LUNG_CANCER']

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the SVM model
    svm_model = SVC(probability=True, random_state=42)
    svm_model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Save the model
    joblib.dump(svm_model, 'models/svm_model.pkl')

    return jsonify({
        "message": "Model trained successfully.",
        "accuracy": accuracy,
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    })


@app.route('/predict', methods=['POST'])
def predict():
    global svm_model

    # Ensure the model is loaded
    if svm_model is None:
        try:
            svm_model = joblib.load('models/svm_model.pkl')
        except FileNotFoundError:
            return jsonify({"error": "Model not found. Train the model first."}), 400

    # Get input data
    input_data = request.get_json()
    features = pd.DataFrame(input_data['features'])

    # Make predictions
    predictions = svm_model.predict(features)
    probabilities = svm_model.predict_proba(features)

    return jsonify({
        "predictions": predictions.tolist(),
        "probabilities": probabilities.tolist()
    })


@app.route('/load_model', methods=['GET'])
def load_model():
    global svm_model

    # Load the saved model
    try:
        svm_model = joblib.load('models/svm_model.pkl')
        return jsonify({"message": "Model loaded successfully."})
    except FileNotFoundError:
        return jsonify({"error": "Model not found. Train the model first."}), 400


@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to the SVM API! Use /train, /predict, or /load_model endpoints."})


if __name__ == '__main__':
    app.run(debug=True)