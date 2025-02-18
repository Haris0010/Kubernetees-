from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
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
XVAL_PATH = os.getenv('XVAL_PATH')
YVAL_PATH = os.getenv('YVAL_PATH')

@app.route('/train', methods=['POST'])
def train_model():
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
        X = df.drop(columns=["resale_price"])
        y = df["resale_price"]

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        X_val.to_csv(XVAL_PATH, index=False)
        y_val.to_csv(YVAL_PATH, index=False)

        models = {
            "RandomForest": RandomForestRegressor(random_state=42),
            "LinearRegression": LinearRegression(),
            "GradientBoosting": GradientBoostingRegressor(random_state=42)
        }
        for model_name, model in models.items():
            model.fit(X, y)
            joblib.dump(model, f"{MODEL_SAVE_PATH}{model_name}.joblib")
        return jsonify({"message": "Training completed!"}), 200
    except Exception as e:
        logging.error(f"Error in training: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=True)
