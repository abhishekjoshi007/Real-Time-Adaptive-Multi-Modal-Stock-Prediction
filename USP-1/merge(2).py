#integratinging microeconomic data with historic data

import pandas as pd
import os

# File paths and configurations
macro_data_file = '/Users/abhishekjoshi/Documents/GitHub/stock_forecasting_CAI/CSV/microeconomic.csv'  # Update with your macroeconomic data file path
historic_data_dir = '/Users/abhishekjoshi/Documents/GitHub/stock_forecasting_CAI/Data 2'  # Update with your historical data directory

# Load macroeconomic data
macro_data = pd.read_csv(macro_data_file)

# Convert macroeconomic data Date column to datetime format
macro_data['Date'] = pd.to_datetime(macro_data['Date'])

print("Macro Data Columns:", macro_data.columns)
print("Macro Data Sample:")
print(macro_data.head())

# Function to process and merge historic data with macroeconomic data
def merge_historic_with_macro(historic_file, macro_data, output_file):
    try:
        # Load historic data
        historic_data = pd.read_csv(historic_file)

        # Convert historic data Date column to datetime format
        historic_data['Date'] = pd.to_datetime(historic_data['Date'])

        print(f"Processing {os.path.basename(historic_file)}...")
        print("Historic Data Columns:", historic_data.columns)
        print("Historic Data Sample:")
        print(historic_data.head())

        # Merge the historic and macroeconomic data on the Date column
        merged_data = historic_data.merge(macro_data, on='Date', how='left')

        # Save the merged data to the output file
        merged_data.to_csv(output_file, index=False)
        print(f"Merged data saved to: {output_file}")
    except Exception as e:
        print(f"Error merging data for {os.path.basename(historic_file)}: {e}")

# Process each ticker's historic data file
for ticker in os.listdir(historic_data_dir):
    ticker_path = os.path.join(historic_data_dir, ticker)
    if os.path.isdir(ticker_path):
        historic_file = os.path.join(ticker_path, f"{ticker}_USP1_features.csv")
        output_file = os.path.join(ticker_path, f"{ticker}_merged.csv")  # Save the output in the ticker's folder

        if os.path.exists(historic_file):
            merge_historic_with_macro(historic_file, macro_data, output_file)
        else:
            print(f"Historic data file missing for {ticker}, skipping...")
