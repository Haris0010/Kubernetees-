from flask import Flask, request, jsonify
from joblib import load, dump
from sklearn.metrics import r2_score
import pandas as pd
import os
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

MODEL_SAVE_PATH = os.getenv('MODEL_SAVE_PATH')

@app.route('/evaluate', methods=['POST'])
def evaluate_models():
    try:
        models = {name: load(f'{MODEL_SAVE_PATH}{name}.joblib') for name in ["RandomForest", "LinearRegression", "GradientBoosting"]}
        X_val = pd.read_csv("/app/data/X_val.csv")
        y_val = pd.read_csv("/app/data/y_val.csv")
        best_model, best_score, best_name = None, -float("inf"), ""
        for name, model in models.items():
            score = r2_score(y_val, model.predict(X_val))
            if score > best_score:
                best_model, best_score, best_name = model, score, name
        dump(best_model, f'{MODEL_SAVE_PATH}saved_model.joblib')
        return jsonify({"message": f"Best model: {best_name}"}), 200
    except Exception as e:
        logging.error(f"Error in evaluation: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8002, debug=True)