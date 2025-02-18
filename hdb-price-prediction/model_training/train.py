from flask import Flask, request, jsonify
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import pandas as pd
import joblib
import os
import requests
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

PROCESSED_DATA_PATH = os.getenv('PROCESSED_DATA_PATH')
MODEL_SAVE_PATH = os.getenv('MODEL_SAVE_PATH')

@app.route('/train', methods=['POST'])
def train_model():
    try:
        if not os.path.exists(PROCESSED_DATA_PATH):
            return jsonify({"error": "Processed data file missing"}), 400

        df = pd.read_csv(PROCESSED_DATA_PATH)

        if df.empty:
            return jsonify({"error": "Processed dataset is empty"}), 400

        X = df.drop(columns=["resale_price"], errors='ignore')
        y = df.get("resale_price")

        models = {
            "RandomForest": RandomForestRegressor(random_state=42),
            "LinearRegression": LinearRegression(),
            "GradientBoosting": GradientBoostingRegressor(random_state=42)
        }

        for model_name, model in models.items():
            try:
                model.fit(X, y)
                joblib.dump(model, f"{MODEL_SAVE_PATH}{model_name}.joblib")
                logging.info(f"{model_name} trained and saved successfully.")
            except Exception as e:
                logging.error(f"Training failed for {model_name}: {e}")

        response = requests.post("http://evaluation-service:8000/evaluate")
        if response.status_code == 200:
            logging.info("Evaluation service called successfully.")
            return jsonify({"message": "Training completed, Evaluation started!"})
        else:
            return jsonify({"error": "Failed to call evaluation service"}), 500

    except Exception as e:
        logging.error(f"Error in training: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
