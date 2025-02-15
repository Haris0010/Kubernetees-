from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib


# File paths
PROCESSED_DATA_PATH = "/app/data/processed_data.csv"
MODEL_SAVE_PATH = "/app/data/"

size_split = 0.2

# Load processed dataset
print(" Loading processed data...")
df = pd.read_csv(PROCESSED_DATA_PATH)

# **Dynamically drop all one-hot encoded 'month_*' columns**
month_columns = [col for col in df.columns if col.startswith("month_")]
df = df.drop(columns=month_columns)

# Define features (X) and target (y)
X = df.drop(columns=["resale_price"])  # Drop target
y = df["resale_price"]

# Split dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=size_split, random_state=42)
X_val.to_csv(f"/app/data/X_val.csv", index=False)
y_val.to_csv(f"/app/data/y_val.csv", index=False)


# Define models to compare
models = {
    "RandomForest": RandomForestRegressor(random_state=42),
    "LinearRegression": LinearRegression(),
    "GradientBoosting": GradientBoostingRegressor(random_state=42)
}

def train(X_train, X_val, y_train, y_val, model, model_name):
    """Train model"""
    
    print(f"\n Training {model_name}...")
    model.fit(X_train, y_train)
    joblib.dump(model, f"{MODEL_SAVE_PATH}{model_name}.joblib")


     


def train_and_evaluate(X_train, X_val, y_train, y_val, model, model_name):
    """Train model and evaluate performance"""
    
    print(f"\n Training {model_name}...")
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)

    # Calculate performance metrics
    mae = mean_absolute_error(y_val, y_val_pred)
    mse = mean_squared_error(y_val, y_val_pred)
    r2 = r2_score(y_val, y_val_pred)

    print(f"\n {model_name} Performance:")
    print(f" - MAE: {mae:.2f}")
    print(f" - MSE: {mse:.2f}")
    print(f" - R² Score: {r2:.4f}")

    return r2, model



def main():
    # best_overall_model = None
    # best_overall_score = -float("inf")
    # best_model_name = ""

    # # Train each model and evaluate performance
    # for model_name, model in models.items():
    #     r2, trained_model = train_and_evaluate(X_train, X_val, y_train, y_val, model, model_name)
    #     if r2 > best_overall_score:
    #         best_overall_score = r2
    #         best_overall_model = trained_model
    #         best_model_name = model_name

    # # Save the best model
    # save_model(best_overall_model, best_model_name, best_overall_score)
    for model_name, model in models.items():
        train(X_train, X_val, y_train, y_val, model, model_name)
        print(f"Training of {model_name} complete")
    print("All Models Completed")

if __name__ == "__main__":
    main()
