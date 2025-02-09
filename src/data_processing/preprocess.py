import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def handle_missing_values(data_path, dataset_name):
    # Load raw data
    df = pd.read_csv(data_path)
    
    # Check missing values
    missing = df.isnull().sum()
    print(f"Missing values in {dataset_name}: \n{missing}\n")
    
    # Impute or drop
    if dataset_name == "Fraud_Data":
        # Example: Impute numerical columns with median
        num_cols = ['age', 'purchase_value']
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        
        # Drop rows with missing categorical data (e.g., 'source')
        df.dropna(subset=['source', 'browser'], inplace=True)
        
    elif dataset_name == "creditcard":
        # Assuming PCA columns (V1-V28) have no missing values
        df['Amount'] = df['Amount'].fillna(df['Amount'].median())
    
    # Save processed data
    output_path = f"data/processed/cleaned_{dataset_name}.csv"
    df.to_csv(output_path, index=False)
    return df

# Run for Fraud_Data.csv
fraud_df = handle_missing_values("data/raw/Fraud_Data.csv", "Fraud_Data")

# Run for creditcard.csv
credit_df = handle_missing_values("data/raw/creditcard.csv", "creditcard")

def clean_data(df, dataset_name):
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Correct data types
    if dataset_name == "Fraud_Data":
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
        
        # Convert IP to integer safely
        df['ip_address'] = (
            df['ip_address']
            .astype(str)  # Force to string to handle NaN/float entries
            .apply(lambda x: int(x.replace('.', '')) if x != 'nan' else None)
        )
        
        # Drop rows with invalid IP addresses (e.g., 'nan')
        df = df.dropna(subset=['ip_address'])
    
    return df

# Example usage
fraud_df = clean_data(fraud_df, "Fraud_Data")


def add_features(df):
    # Time-based features
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek
    
    # Transaction frequency per user
    df['transaction_count'] = df.groupby('user_id')['user_id'].transform('count')
    
    return df

fraud_df = add_features(fraud_df)



def normalize_data(df):
    scaler = StandardScaler()
    num_cols = ['purchase_value', 'age', 'transaction_count']
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df

fraud_df = normalize_data(fraud_df)

def encode_categorical(df):
 available_cols = [col for col in ['source', 'browser', 'sex', 'country'] if col in df.columns]
 df = pd.get_dummies(df, columns=available_cols)

fraud_df = encode_categorical(fraud_df)