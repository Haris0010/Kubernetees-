from flask import Flask, request, jsonify
from joblib import load
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import os
import requests

app = Flask(__name__)

MODEL_SAVE_PATH = os.getenv('MODEL_SAVE_PATH')

@app.route('/evaluate', methods=['POST'])
def evaluate_models():
    models = {
        "RandomForest": load(f'{MODEL_SAVE_PATH}RandomForest.joblib'),
        "LinearRegression": load(f'{MODEL_SAVE_PATH}LinearRegression.joblib'),
        "GradientBoosting": load(f'{MODEL_SAVE_PATH}GradientBoosting.joblib')
    }

    X_val = pd.read_csv("/app/data/X_val.csv")
    y_val = pd.read_csv("/app/data/y_val.csv")

    best_model = None
    best_score = -float("inf")
    best_model_name = ""

    for model_name, model in models.items():
        y_pred = model.predict(X_val)
        r2 = r2_score(y_val, y_pred)
        if r2 > best_score:
            best_score = r2
            best_model = model
            best_model_name = model_name

    # Save best model
    joblib.dump(best_model, f'{MODEL_SAVE_PATH}saved_model.joblib')

    # Call Prediction API
    requests.post("http://prediction-service:8000/predict")

    return jsonify({"message": f"Evaluation completed, best model is {best_model_name}. Prediction started!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)