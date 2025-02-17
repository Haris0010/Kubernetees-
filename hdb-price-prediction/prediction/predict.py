from flask import Flask, request, jsonify
from joblib import load
import pandas as pd
import os

app = Flask(__name__)

PROCESSED_DATA_PATH = os.getenv('PROCESSED_DATA_PATH')
PRED_DATA_PATH = os.getenv('PRED_DATA_PATH')
MODEL_SAVE_PATH = os.getenv('MODEL_SAVE_PATH')

@app.route('/predict', methods=['POST'])
def predict():
    model = load(f'{MODEL_SAVE_PATH}saved_model.joblib')
    test = pd.read_csv(PRED_DATA_PATH)
    train = pd.read_csv(PROCESSED_DATA_PATH)
    
    train_columns = train.drop(columns=['resale_price']).columns
    test = test[train_columns]
    
    predictions = model.predict(test)
    test['Predictions'] = predictions
    test.to_csv(f'{MODEL_SAVE_PATH}predictions_output.csv', index=False)

    # Send results back to Flask API
    requests.post("http://flask-api-service:5000/results", json={"predictions": predictions.tolist()})

    return jsonify({"message": "Prediction completed, results sent to Flask API."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
