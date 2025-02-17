from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import os

PROCESSED_DATA_PATH = os.getenv('PROCESSED_DATA_PATH')
MODEL_SAVE_PATH = os.getenv('MODEL_SAVE_PATH')

size_split = 0.2
print("Loading processed data...")
df = pd.read_csv(PROCESSED_DATA_PATH)

month_columns = [col for col in df.columns if col.startswith("month_")]
df = df.drop(columns=month_columns)
X = df.drop(columns=["resale_price"])
y = df["resale_price"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=size_split, random_state=42)
X_val.to_csv("/app/data/X_val.csv", index=False)
y_val.to_csv("/app/data/y_val.csv", index=False)

models = {
    "RandomForest": RandomForestRegressor(random_state=42),
    "LinearRegression": LinearRegression(),
    "GradientBoosting": GradientBoostingRegressor(random_state=42)
}

def train(X_train, X_val, y_train, y_val, model, model_name):
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    joblib.dump(model, f"{MODEL_SAVE_PATH}{model_name}.joblib")
def main():
    for model_name, model in models.items():
        train(X_train, X_val, y_train, y_val, model, model_name)
        print(f"Training of {model_name} complete")
    print("All Models Completed")

if __name__ == "__main__":
    main()
