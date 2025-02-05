from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import joblib

PROCESSED_DATA_PATH = "hdb-price-prediction/data/processed_data.csv"
saved_model_path = "hdb-price-prediction/saved_model"
size_split = 0.2
cv = 5

data = pd.read_csv(PROCESSED_DATA_PATH)

# Define models to compare
RandomForest = RandomForestClassifier(random_state=42)
LogisticRegression = LogisticRegression(max_iter=1000, random_state=42)
GradientBoosting = GradientBoostingClassifier(random_state=42)

# Define hyperparameter grids for tuning
param_grids = {
    "RandomForest": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5]
    },
    "LogisticRegression": {
        "C": [0.01, 0.1, 1, 10],
        "solver": ["lbfgs", "liblinear"]
    },
    "GradientBoosting": {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7]
    }
}

def split_data(train_data):
    X = train_data.drop(columns=["resale_price"])
    y = train_data["resale_price"]

    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=size_split, random_state=42)
    return X_train, X_val, y_train, y_val


def training(X_train, X_val, y_train, y_val, model, model_name):
    

    # Train model
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    y_val_prob = model.predict_proba(X_val)[:, 1]
    accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Classification Report:\n", classification_report(y_val, y_val_pred))
    print(f"Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred))
    roc_auc = roc_auc_score(y_val, y_val_prob)
    print(f"ROC AUC Score: {roc_auc}")

    # Cross-validation scores
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv)
    cv_mean_accuracy = np.mean(cv_scores)
    print(f"Cross-Validation Accuracy: {cv_mean_accuracy:.4f} ± {np.std(cv_scores):.4f}")

    return roc_auc, model

def saveModel(best_model, best_model_name, best_accuracy):
    # Save the best model
    joblib.dump(best_model, f'{saved_model_path}/best_model.joblib')
    print(f"\nBest Model: {best_model_name} with ROC AUC Score: {best_accuracy:.4f}")
    print(f"{best_model_name} model saved successfully!")

def main():
    print("Step 1: Split the Data ")
    X_train, X_val, y_train, y_val = split_data(data)
    LogisticRegression.fit(X_train, y_train)
    y_pred = LogisticRegression.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    joblib.dump(LogisticRegression, f'{saved_model_path}/best_model.joblib')


    
    # print("Step 2: Train the Models")
    # model_objects = {"RandomForest": RandomForest, "LogisticRegression": LogisticRegression, "GradientBoosting": GradientBoosting}
    
    # ModelRocAuc = {}
    # TunedModels = {}
    # for model_name, model in model_objects.items():
    #     print(f"\n{model_name}")
    #     roc_auc, tuned_model = training(X_train, X_val, y_train, y_val, model, model_name)
    #     ModelRocAuc[model_name] = roc_auc
    #     TunedModels[model_name] = tuned_model

    # # Find the best model
    # BestModelName, BestRocAuc = max(ModelRocAuc.items(), key=lambda x: x[1])
    # BestModel = TunedModels[BestModelName]

    # # Save the best model
    # saveModel(BestModel, BestModelName, BestRocAuc)

if __name__ == "__main__":
    main()  