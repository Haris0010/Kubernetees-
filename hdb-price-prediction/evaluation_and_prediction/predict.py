from joblib import load
import configparser
from pathlib import Path
import pandas as pd

PROCESSED_DATA_PATH = "hdb-price-prediction/data/processed_data.csv"
saved_model_path = "hdb-price-prediction/saved_model"
test = pd.read_csv(f'test_dataset')
train = pd.read_csv(PROCESSED_DATA_PATH)



model = load(f'{saved_model_path}/best_model.joblib')



train_columns = train.drop(columns=['resale_price']).columns
test = test[train_columns]


predictions = model.predict(test)

test['Predictions'] = predictions

# Save to a new CSV file
test.to_csv(f'{prediction_data_path}/predictions_output.csv', index=False)

