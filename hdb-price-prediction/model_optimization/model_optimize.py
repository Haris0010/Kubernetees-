from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import joblib

# File paths
PROCESSED_DATA_PATH = "data/processed_data.csv"
OPTIMIZED_MODEL_PATH = "data/optimized_model.joblib"
size_split = 0.2

# Load processed dataset
print("Loading processed data...")
df = pd.read_csv(PROCESSED_DATA_PATH)

# **Dynamically drop all one-hot encoded 'month_*' columns**
month_columns = [col for col in df.columns if col.startswith("month_")]
df = df.drop(columns=month_columns)

# Define features (X) and target (y)
X = df.drop(columns=["resale_price"])  # Drop target
y = df["resale_price"]

# Split dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=size_split, random_state=42)

# Define models and hyperparameter grids
param_grids = {
    "RandomForest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10]
    },
    "GradientBoosting": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7]
    }
}

models = {
    "RandomForest": RandomForestRegressor(random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42)
}

def tune_model(model, param_grid, model_name):
    """Perform GridSearchCV for hyperparameter tuning"""
    print(f"\n Optimizing {model_name}...")

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    # Get the best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"\n Best {model_name} Model:")
    print(f" - Best Parameters: {best_params}")
    print(f" - Best R² Score: {best_score:.4f}")

    return best_model, best_score

def main():
    best_overall_model = None
    best_overall_score = -float("inf")
    best_model_name = ""

    # Tune each model
    for model_name, model in models.items():
        best_model, best_score = tune_model(model, param_grids[model_name], model_name)

        # Save the best performing model
        if best_score > best_overall_score:
            best_overall_score = best_score
            best_overall_model = best_model
            best_model_name = model_name

    # Save the best optimized model
    joblib.dump(best_overall_model, OPTIMIZED_MODEL_PATH)
    print(f"\n Best Optimized Model: {best_model_name} with R² Score: {best_overall_score:.4f}")
    print(f" Optimized model saved at {OPTIMIZED_MODEL_PATH}")

if __name__ == "__main__":
    main()
