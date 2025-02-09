import pandas as pd

def merge_geolocation():
    # Load processed Fraud_Data and geolocation data
    fraud_df = pd.read_csv("data/processed/cleaned_Fraud_Data.csv")
    ip_df = pd.read_csv("data/geolocation/IpAddress_to_Country.csv")
    
    # Convert IP ranges to integers
    ip_df['lower_bound'] = ip_df['lower_bound_ip_address'].astype(int)
    ip_df['upper_bound'] = ip_df['upper_bound_ip_address'].astype(int)
    
    # Convert ip_address in fraud_df to int (after handling NaN values)
    fraud_df['ip_address'] = fraud_df['ip_address'].fillna(0).astype(int)

    # Merge using conditional join (Pandas 2.0+)
    merged_df = pd.merge_asof(
        fraud_df.sort_values('ip_address'),
        ip_df.sort_values('lower_bound'),
        left_on='ip_address',
        right_on='lower_bound',
        direction='backward'
    )
    
    # Save merged data
    merged_df.to_csv("data/processed/fraud_data_with_country.csv", index=False)
    return merged_df

merge_geolocation()
