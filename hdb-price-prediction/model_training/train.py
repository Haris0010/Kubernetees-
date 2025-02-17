from flask import Flask, request, jsonify
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import pandas as pd
import joblib
import os

app = Flask(__name__)

PROCESSED_DATA_PATH = os.getenv('PROCESSED_DATA_PATH')
MODEL_SAVE_PATH = os.getenv('MODEL_SAVE_PATH')

@app.route('/train', methods=['POST'])
def train_model():
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

    # Call Evaluation API
    requests.post("http://evaluation-service:8000/evaluate")

    return jsonify({"message": "Training completed, Evaluation started!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)