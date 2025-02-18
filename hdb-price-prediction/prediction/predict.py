from flask import Flask, request, jsonify
from joblib import load
import pandas as pd
import os
import requests
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

PROCESSED_DATA_PATH = os.getenv('PROCESSED_DATA_PATH')
PRED_DATA_PATH = os.getenv('PRED_DATA_PATH')
MODEL_SAVE_PATH = os.getenv('MODEL_SAVE_PATH')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        model = load(f'{MODEL_SAVE_PATH}saved_model.joblib')

        if not os.path.exists(PROCESSED_DATA_PATH) or not os.path.exists(PRED_DATA_PATH):
            return jsonify({"error": "Processed or prediction data files missing"}), 400

        test = pd.read_csv(PRED_DATA_PATH)
        train = pd.read_csv(PROCESSED_DATA_PATH)

        if test.empty or train.empty:
            return jsonify({"error": "One or both datasets are empty"}), 400

        train_columns = train.drop(columns=['resale_price'], errors='ignore').columns
        test = test[train_columns]

        predictions = model.predict(test)
        test['Predictions'] = predictions
        test.to_csv(f'{MODEL_SAVE_PATH}predictions_output.csv', index=False)
        logging.info("Predictions saved successfully.")

        response = requests.post("http://flask-api-service:5000/results", json={"predictions": predictions.tolist()})
        if response.status_code == 200:
            logging.info("Results sent to Flask API successfully.")
            return jsonify({"message": "Prediction completed, results sent to Flask API."})
        else:
            return jsonify({"error": "Failed to send results to Flask API"}), 500

    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)