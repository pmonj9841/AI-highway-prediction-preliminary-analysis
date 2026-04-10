import pandas as pd
import os

def main():
    # File paths
    input_path = 'data/processed/data_travel_time.csv'
    output_path = 'data/processed/data_travel_time_xgboost.csv'
    
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found.")
        return

    # Load data
    print(f"Loading data from {input_path}...")
    # Using utf-8 as it seemed to be the most promising, 
    # and errors during reading often happen due to specific characters.
    try:
        df = pd.read_csv(input_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(input_path, encoding='cp949')

    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by route and date for correct lag calculation
    df = df.sort_values(['gate_start', 'gate_end', 'date'])
    
    # Define lags
    lags = [1, 2, 3, 24, 168]
    
    print("Creating lag features...")
    # Group by route and apply shift
    for lag in lags:
        df[f'travel_time(t-{lag})'] = df.groupby(['gate_start', 'gate_end'])['travel_time'].shift(lag)
    
    # Drop rows with NaN values resulting from lags
    # (Optional: XGBoost can handle NaNs, but for 168 lag we lose first week)
    # The user didn't explicitly ask to drop, but it's common practice.
    df = df.dropna()
    
    # Save processed data
    print(f"Saving processed data to {output_path}...")
    df.to_csv(output_path, index=False, encoding='utf-8-sig') # utf-8-sig for Excel compatibility with Korean
    print("Preprocessing completed successfully.")

if __name__ == "__main__":
    main()
