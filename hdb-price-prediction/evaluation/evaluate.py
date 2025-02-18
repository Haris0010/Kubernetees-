from flask import Flask, request, jsonify
from joblib import load, dump
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import os
import requests
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

MODEL_SAVE_PATH = os.getenv('MODEL_SAVE_PATH')

@app.route('/evaluate', methods=['POST'])
def evaluate_models():
    try:
        models = {
            "RandomForest": load(f'{MODEL_SAVE_PATH}RandomForest.joblib'),
            "LinearRegression": load(f'{MODEL_SAVE_PATH}LinearRegression.joblib'),
            "GradientBoosting": load(f'{MODEL_SAVE_PATH}GradientBoosting.joblib')
        }

        X_val = pd.read_csv("/app/data/X_val.csv")
        y_val = pd.read_csv("/app/data/y_val.csv")

        if X_val.empty or y_val.empty:
            return jsonify({"error": "Validation datasets are empty"}), 400

        best_model = None
        best_score = -float("inf")
        best_model_name = ""

        for model_name, model in models.items():
            try:
                y_pred = model.predict(X_val)
                r2 = r2_score(y_val, y_pred)
                logging.info(f"{model_name} R2 Score: {r2}")
                if r2 > best_score:
                    best_score = r2
                    best_model = model
                    best_model_name = model_name
            except Exception as e:
                logging.error(f"Evaluation failed for {model_name}: {e}")

        dump(best_model, f'{MODEL_SAVE_PATH}saved_model.joblib')
        logging.info(f"Best model ({best_model_name}) saved.")

        response = requests.post("http://prediction-service:8000/predict")
        if response.status_code == 200:
            logging.info("Prediction service called successfully.")
            return jsonify({"message": f"Evaluation completed, best model is {best_model_name}. Prediction started!"})
        else:
            return jsonify({"error": "Failed to call prediction service"}), 500

    except Exception as e:
        logging.error(f"Error in evaluation: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)