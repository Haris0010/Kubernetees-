from joblib import load

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import joblib

PROCESSED_DATA_PATH = "hdb-price-prediction/data/processed_data.csv"
saved_model_path = "hdb-price-prediction/saved_model"


model = load(f'{saved_model_path}/best_model.joblib')

def eval(X_val,y_val, model, model_name):
    y_val_pred = model.predict(X_val)
    y_val_prob = model.predict_proba(X_val)[:, 1]
    accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Classification Report:\n", classification_report(y_val, y_val_pred))
    print(f"Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred))
    roc_auc = roc_auc_score(y_val, y_val_prob)
    print(f"ROC AUC Score: {roc_auc}")

    return roc_auc, model

train_columns = train.drop(columns=['Survived']).columns
test = test[train_columns]


predictions = model.predict(test)

test['Predictions'] = predictions

# Save to a new CSV file
test.to_csv(f'{prediction_data_path}/predictions_output.csv', index=False)

