import pandas as pd
from sklearn.preprocessing import LabelEncoder


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
    """ Clean dataset dynamically: handle missing values, convert columns, remove outliers """
    print("Cleaning data...")

    # Drop unnecessary columns if they exist
    drop_cols = ["block", "street_name"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # Convert 'month' to datetime format and extract year
    if "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"], format="%Y-%m").dt.strftime("%Y-%m")
        df["year"] = pd.to_datetime(df["month"], format="%Y-%m").dt.year

    # Convert 'storey_range' to numeric (extract median value)
    if "storey_range" in df.columns:
        df[["storey_min", "storey_max"]] = df["storey_range"].str.extract(r"(\d+)\s*TO\s*(\d+)").astype(float)
        df["storey_range"] = df[["storey_min", "storey_max"]].mean(axis=1)
        df = df.drop(columns=["storey_min", "storey_max"])  # Drop temp columns

    # Convert 'remaining_lease' into numeric values (extract years)
    if "remaining_lease" in df.columns:
        df["remaining_lease"] = df["remaining_lease"].str.extract(r"(\d+)").astype(float)

    # **Dynamically detect categorical and numerical columns**
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()

    # Drop rows with missing categorical values (no hardcoded names)
    df.dropna(subset=categorical_cols, inplace=True)

    # Fill missing numerical values with median
    for col in numerical_cols:
        df[col].fillna(df[col].median(), inplace=True)

    # Remove extreme outliers (IQR method) for numerical columns
    for col in numerical_cols:
        if col not in ["resale_price", "year"]:  # Exclude target variable & year
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df[col] >= (Q1 - 1.5 * IQR)) & (df[col] <= (Q3 + 1.5 * IQR))]

    # One-hot encoding for categorical variables
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Convert boolean values to 0/1 for ML models, excluding 'month'
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

