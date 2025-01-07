import os
import pandas as pd

# Define the root directory where your data is stored
root_directory = "/Users/abhishekjoshi/Documents/GitHub/stock_forecasting_CAI/Data 2"

# Loop through each ticker folder in the root directory
for ticker_folder in os.listdir(root_directory):
    ticker_path = os.path.join(root_directory, ticker_folder)
    
    # Skip if not a directory
    if not os.path.isdir(ticker_path):
        continue
    
    # Construct the input file path for the ticker
    input_file_path = os.path.join(ticker_path, f"{ticker_folder}_prepared_data.csv")
    if not os.path.exists(input_file_path):
        print(f"Prepared data file not found for ticker {ticker_folder}: {input_file_path}")
        continue
    
    # Load the dataset
    data = pd.read_csv(input_file_path)
    
    # Ensure 'Volume-Weighted Sentiment' column exists
    if 'Volume-Weighted Sentiment' not in data.columns:
        print(f"'Volume-Weighted Sentiment' column is missing for ticker {ticker_folder}. Skipping...")
        continue

    # Step 1: Calculate rolling mean and standard deviation
    rolling_mean = data['Volume-Weighted Sentiment'].rolling(window=7).mean()
    rolling_std = data['Volume-Weighted Sentiment'].rolling(window=7).std()

    # Step 2: Compute Normalized VWS (Z-Score Normalization)
    data['Normalized VWS (7 Days)'] = (
        data['Volume-Weighted Sentiment'] - rolling_mean
    ) / rolling_std

    # Step 3: Handle NaN values for the first 6 rows
    data['Normalized VWS (7 Days)'].fillna(0, inplace=True)

    # Debug: Verify calculations
    print(f"Normalized VWS (7 Days) calculated for {ticker_folder}:")
    print(data[['Volume-Weighted Sentiment', 'Normalized VWS (7 Days)']].head(10))

    # Save the corrected data for the ticker in the same folder as the input file
    output_file_path = os.path.join(ticker_path, f"{ticker_folder}_final.csv")
    data.to_csv(output_file_path, index=False)

    print(f"Corrected data saved for {ticker_folder} to {output_file_path}.")
