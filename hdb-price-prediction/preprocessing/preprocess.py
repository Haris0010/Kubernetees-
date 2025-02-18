from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import requests
import logging

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load paths from environment variables
RAW_DATA_PATH = os.getenv('RAW_DATA_PATH')
RAW_PREDICTION_PATH = os.getenv('RAW_PREDICTION_PATH')
PROCESSED_DATA_PATH = os.getenv('PROCESSED_DATA_PATH')
PRED_DATA_PATH = os.getenv('PRED_DATA_PATH')

def clean_data(df):
    try:
        drop_cols = ["block", "street_name"]
        df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

        if "month" in df.columns:
            df["month"] = pd.to_datetime(df["month"]).dt.month
            df["year"] = pd.to_datetime(df["month"]).dt.year

        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
        numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()

        for col in numerical_cols:
            df[col].fillna(df[col].median(), inplace=True)

        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

        return df
    except Exception as e:
        logging.error(f"Error cleaning data: {e}")
        raise

@app.route('/process', methods=['POST'])
def process_data():
    try:
        if not os.path.exists(RAW_DATA_PATH) or not os.path.exists(RAW_PREDICTION_PATH):
            return jsonify({"error": "Input files missing"}), 400

        df_train = pd.read_csv(RAW_DATA_PATH)
        df_pred = pd.read_csv(RAW_PREDICTION_PATH)

        if df_train.empty or df_pred.empty:
            return jsonify({"error": "One or both datasets are empty"}), 400

        df_train = clean_data(df_train)
        df_pred = clean_data(df_pred)

        df_train.to_csv(PROCESSED_DATA_PATH, index=False)
        df_pred.to_csv(PRED_DATA_PATH, index=False)

        response = requests.post("http://training-service:8000/train")
        if response.status_code == 200:
            logging.info("Training service called successfully.")
            return jsonify({"message": "Preprocessing completed, Training started!"})
        else:
            return jsonify({"error": "Failed to call training service"}), 500
    except Exception as e:
        logging.error(f"Error in processing: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)