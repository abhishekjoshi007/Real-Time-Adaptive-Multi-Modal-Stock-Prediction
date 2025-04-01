import pandas as pd
import os

# File paths
data_dir = '/Users/abhishekjoshi/Documents/GitHub/stock_forecasting_CAI/Data 2'  # Main data directory containing ticker folders
vix_file = '/Users/abhishekjoshi/Documents/GitHub/stock_forecasting_CAI/CSV/VIX_History.csv'  # Path to your VIX data file

# Load VIX data
vix_data = pd.read_csv(vix_file)
vix_data['DATE'] = pd.to_datetime(vix_data['DATE'])  # Ensure DATE column is in datetime format
vix_data = vix_data.rename(columns={'DATE': 'Date'})  # Rename to match historic data column

print("VIX Data Columns:", vix_data.columns)
print("Sample VIX Data:")
print(vix_data.head())

# Iterate through each ticker folder
for ticker in os.listdir(data_dir):
    ticker_path = os.path.join(data_dir, ticker)
    if os.path.isdir(ticker_path):
        historic_file = os.path.join(ticker_path, f"{ticker}_merged.csv")
        output_file = os.path.join(ticker_path, f"{ticker}_merged_with_vix.csv")

        if os.path.exists(historic_file):
            try:
                print(f"Processing {ticker}...")

                # Load historic data
                historic_data = pd.read_csv(historic_file)
                historic_data['Date'] = pd.to_datetime(historic_data['Date'])  # Ensure Date column is in datetime format

                # Merge with VIX data
                merged_data = historic_data.merge(vix_data, on='Date', how='left')

                # Save the merged file
                merged_data.to_csv(output_file, index=False)
                print(f"Merged file saved for {ticker} at {output_file}")
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
        else:
            print(f"Historic data file missing for {ticker}, skipping...")
