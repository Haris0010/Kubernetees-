from joblib import load
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import os

MODEL_SAVE_PATH = os.getenv('MODEL_SAVE_PATH')

GB = load(f'{MODEL_SAVE_PATH}GradientBoosting.joblib')
LR = load(f'{MODEL_SAVE_PATH}LinearRegression.joblib')
RF = load(f'{MODEL_SAVE_PATH}RandomForest.joblib')

models = {
    "RandomForest": RF,
    "LinearRegression": LR,
    "GradientBoosting": GB
}

X_val = pd.read_csv("/app/data/X_val.csv")
y_val = pd.read_csv("/app/data/y_val.csv")

def eval(X_val, y_val, model, model_name):
    y_val_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_val_pred)
    mse = mean_squared_error(y_val, y_val_pred)
    r2 = r2_score(y_val, y_val_pred)
    print(f"\n{model_name} Performance:")
    print(f" - MAE: {mae:.2f}")
    print(f" - MSE: {mse:.2f}")
    print(f" - R² Score: {r2:.4f}")
    return r2, model

def save_model(best_model, best_model_name, best_r2_score):
    import joblib
    joblib.dump(best_model, MODEL_SAVE_PATH)
    print(f"\nBest Model: {best_model_name} with R² Score: {best_r2_score:.4f}")
    print(f"{best_model_name} model saved successfully at {MODEL_SAVE_PATH}!")

def main():
    best_overall_model = None
    best_overall_score = -float("inf")
    best_model_name = ""
    for model_name, model in models.items():
        r2, trained_model = eval(X_val, y_val, model, model_name)
        if r2 > best_overall_score:
            best_overall_score = r2
            best_overall_model = trained_model
            best_model_name = model_name
    save_model(best_overall_model, best_model_name, best_overall_score)
    print("All Models Completed")

if __name__ == "__main__":
    main()
