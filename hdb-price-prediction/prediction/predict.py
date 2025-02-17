from joblib import load
import os
import pandas as pd

# Load paths from environment variables set by the ConfigMap
PROCESSED_DATA_PATH = os.getenv('PROCESSED_DATA_PATH')
PRED_DATA_PATH = os.getenv('PRED_DATA_PATH')
MODEL_SAVE_PATH = os.getenv('MODEL_SAVE_PATH')

# Read data
test = pd.read_csv(PRED_DATA_PATH)
train = pd.read_csv(PROCESSED_DATA_PATH)

# Load the trained model
model = load(f'{MODEL_SAVE_PATH}saved_model.joblib')

# Ensure the test data has the same features as training data
train_columns = train.drop(columns=['resale_price']).columns
test = test[train_columns]

# Make predictions
predictions = model.predict(test)
test['Predictions'] = predictions

# Save predictions
test.to_csv(f'{MODEL_SAVE_PATH}predictions_output.csv', index=False)

print("Predictions saved successfully at /app/data/predictions_output.csv")
