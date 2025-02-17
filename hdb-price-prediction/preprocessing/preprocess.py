import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

RAW_DATA_PATH = os.getenv('RAW_DATA_PATH')
RAW_PREDICTION_PATH = os.getenv('RAW_PREDICTION_PATH')
PROCESSED_DATA_PATH = os.getenv('PROCESSED_DATA_PATH')
PRED_DATA_PATH = os.getenv('PRED_DATA_PATH')

def load_data(path):
    print("Loading raw data...")
    try:
        dfer = pd.read_csv(path)
        print(f"Loaded {dfer.shape[0]} rows and {dfer.shape[1]} columns.")
        return dfer
    except FileNotFoundError:
        print(f"Error: The file at {path} was not found.")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def clean_data(df):
    print("Cleaning data...")
    drop_cols = ["block", "street_name"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    if "month" in df.columns:
        time = pd.to_datetime(df["month"], format="%Y-%m")
        df["month"] = time.dt.month
        df["year"] = time.dt.year
    if "storey_range" in df.columns:
        df[["storey_min", "storey_max"]] = df["storey_range"].str.extract(r"(\d+)\s*TO\s*(\d+)").astype(float)
        df["storey_range"] = df[["storey_min", "storey_max"]].mean(axis=1)
        df = df.drop(columns=["storey_min", "storey_max"])
    if "remaining_lease" in df.columns:
        df["remaining_lease"] = df["remaining_lease"].str.extract(r"(\d+)").astype(float)
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()
    df.dropna(subset=categorical_cols, inplace=True)
    for col in numerical_cols:
        df[col].fillna(df[col].median(), inplace=True)
    for col in numerical_cols:
        if col not in ["resale_price", "year"]:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df[col] >= (Q1 - 1.5 * IQR)) & (df[col] <= (Q3 + 1.5 * IQR))]
    label_encoders = {}
    for col in categorical_cols:
        print(f"Label Encoding {col}...")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    print("Data cleaning completed.")
    return df

def save_data(df, df1):
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Processed data saved to {PROCESSED_DATA_PATH}")
    df1.to_csv(PRED_DATA_PATH, index=False)
    print(f"Prediction data saved to {PRED_DATA_PATH}")

def main():
    df_train = load_data(RAW_DATA_PATH)
    df_pred = load_data(RAW_PREDICTION_PATH)
    df_train = clean_data(df_train)
    df_pred = clean_data(df_pred)
    save_data(df_train, df_pred)

if __name__ == "__main__":
    main()
