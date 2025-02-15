from joblib import load
import configparser
from pathlib import Path
import pandas as pd

PROCESSED_DATA_PATH = "/app/data/processed_data.csv"
PRED_DATA_PATH = "/app/data/pred_data.csv"
test = pd.read_csv(PRED_DATA_PATH)
train = pd.read_csv(PROCESSED_DATA_PATH)

model = load(f'data/saved_model.joblib')

train_columns = train.drop(columns=['resale_price']).columns
test = test[train_columns]


predictions = model.predict(test)

test['Predictions'] = predictions

# Save to a new CSV file
test.to_csv(f'/app/data/predictions_output.csv', index=False)

