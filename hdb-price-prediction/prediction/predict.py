from flask import Flask, request, jsonify
from joblib import load
import pandas as pd
import os
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
        test = pd.read_csv(PRED_DATA_PATH)
        train = pd.read_csv(PROCESSED_DATA_PATH)
        train_columns = train.drop(columns=['resale_price']).columns
        test = test[train_columns]
        predictions = model.predict(test)
        test['Predictions'] = predictions
        test.to_csv(f'{MODEL_SAVE_PATH}predictions_output.csv', index=False)
        return jsonify({"message": "Prediction completed!"}), 200
    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_predictions', methods=['GET'])
def get_predictions():
    try:
        pred_file = pd.read_csv(f'{MODEL_SAVE_PATH}predictions_output.csv')
        return jsonify({"predictions": pred_file['Predictions'].tolist()}), 200
    except Exception as e:
        logging.error(f"Error fetching predictions: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8003, debug=True)