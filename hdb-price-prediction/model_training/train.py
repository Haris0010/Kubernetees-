from flask import Flask, request, jsonify
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import pandas as pd
import joblib
import os
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

PROCESSED_DATA_PATH = os.getenv('PROCESSED_DATA_PATH')
MODEL_SAVE_PATH = os.getenv('MODEL_SAVE_PATH')

@app.route('/train', methods=['POST'])
def train_model():
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
        X = df.drop(columns=["resale_price"])
        y = df["resale_price"]
        models = {
            "RandomForest": RandomForestRegressor(random_state=42),
            "LinearRegression": LinearRegression(),
            "GradientBoosting": GradientBoostingRegressor(random_state=42)
        }
        for model_name, model in models.items():
            model.fit(X, y)
            joblib.dump(model, f"{MODEL_SAVE_PATH}{model_name}.joblib")
        return jsonify({"message": "Training completed!"})
    except Exception as e:
        logging.error(f"Error in training: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
