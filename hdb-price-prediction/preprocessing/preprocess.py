import pandas as pd

# Define file paths
RAW_DATA_PATH = "data/sg-resale-flat-prices.csv"
PROCESSED_DATA_PATH = "data/processed_data.csv"

def load_data():
    """ Load raw dataset """
    print("Loading raw data...")
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

def clean_data(df):
    """ Clean dataset (handle missing values, format columns) """
    print("Cleaning data...")

    # Drop unnecessary columns
    df = df.drop(columns=["block", "street_name"])

    # Convert 'month' to datetime format and extract year
    df["month"] = pd.to_datetime(df["month"], format="%Y-%m")
    df["year"] = df["month"].dt.year

    # Convert 'storey_range' to a numeric median value
    df["storey_range"] = df["storey_range"].str.extract(r"(\d+)").astype(float)

    # Convert 'remaining_lease' into numeric values (extract years)
    df["remaining_lease"] = df["remaining_lease"].str.extract(r"(\d+)").astype(float)

    # One-hot encoding categorical columns
    df = pd.get_dummies(df, columns=["flat_model", "flat_type", "town"], drop_first=True)

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
