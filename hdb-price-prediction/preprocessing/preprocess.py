import pandas as pd
from sklearn.preprocessing import LabelEncoder


# Define file paths
RAW_DATA_PATH = "hdb-price-prediction/data/sg-resale-flat-prices.csv"
PROCESSED_DATA_PATH = "hdb-price-prediction/data/processed_data.csv"

def load_data():
    """ Load raw dataset """
    print("Loading raw data...")
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

def clean_data(df):
    """ Clean dataset (handle missing values, format columns) """
    print(" Cleaning data...")

    # Drop unnecessary columns
    df = df.drop(columns=["block", "street_name"])

    # Drop Null Values
    df = df.dropna()

    # Convert 'month' to datetime format and extract year
    df["month"] = pd.to_datetime(df["month"], format="%Y-%m").dt.strftime("%Y-%m")
    df["year"] = pd.to_datetime(df["month"], format="%Y-%m").dt.year


    # Convert 'storey_range' to a numeric median value
    df[["storey_min", "storey_max"]] = df["storey_range"].str.extract(r"(\d+)\s*TO\s*(\d+)").astype(float)
    df["storey_range"] = df[["storey_min", "storey_max"]].mean(axis=1)
    df = df.drop(columns=["storey_min", "storey_max"])  # Drop temp columns

    # Convert 'remaining_lease' into numeric values (extract years)
    df["remaining_lease"] = df["remaining_lease"].str.extract(r"(\d+)").astype(float)
    
    le = LabelEncoder()
    df["month"] = le.fit_transform(df["month"])
    df["flat_model"] = le.fit_transform(df["flat_model"])
    df["town"] = le.fit_transform(df["town"])


    # One-hot encoding categorical columns
    df = pd.get_dummies(df, columns=["flat_type"], drop_first=True)

    # Convert boolean values to 0/1 for ML models
    df.loc[:, df.columns != "month"] = df.loc[:, df.columns != "month"].astype(int)

    print("Data cleaning completed.")
    return df


def save_data(df):
    """ Save the cleaned dataset """
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Processed data saved to {PROCESSED_DATA_PATH}")

def main():
    df = load_data()
    df = clean_data(df)
    save_data(df)

if __name__ == "__main__":
    main()
