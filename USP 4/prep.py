import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Define the root directory where your data is stored
root_directory = "/Users/abhishekjoshi/Documents/GitHub/stock_forecasting_CAI/Data 2"

# Loop through each ticker folder in the root directory
for ticker_folder in os.listdir(root_directory):
    ticker_path = os.path.join(root_directory, ticker_folder)
    
    # Skip if not a directory
    if not os.path.isdir(ticker_path):
        continue
    
    # Construct the input file path for the ticker
    input_file_path = os.path.join(ticker_path, f"{ticker_folder}_merged_with_vix.csv")
    if not os.path.exists(input_file_path):
        print(f"File not found for ticker {ticker_folder}: {input_file_path}")
        continue
    
    # Load the dataset
    data = pd.read_csv(input_file_path)
    
    # Debug: Confirm column names
    print(f"Processing {ticker_folder} - Columns in the dataset: {data.columns}")

    # Step 1: Retain Relevant Columns
    columns_to_keep = [
        'Date', 'Industry', 'Ticker Name', 'Close', 'Momentum (7 Days)', 
        'Volatility (7 Days)', 'EWMA Volatility', 
        'Volume-Weighted Sentiment', 'Normalized VWS (7 Days)'
    ]
    data = data[columns_to_keep]

    # Step 2: Handle Missing Data
    data.fillna(method='ffill', inplace=True)  # Forward-fill missing data
    data.fillna(method='bfill', inplace=True)  # Backward-fill missing data

    # Step 3: Normalize or Standardize Inputs
    columns_to_normalize = ['Close', 'Momentum (7 Days)', 'Volatility (7 Days)', 
                            'EWMA Volatility', 'Volume-Weighted Sentiment']
    scaler = StandardScaler()
    data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])

    # Step 4: Recalculate Normalized VWS (7 Days)
    # Calculate rolling mean and standard deviation for Volume-Weighted Sentiment
    data['VWS_Mean_7'] = data['Volume-Weighted Sentiment'].rolling(window=7).mean()
    data['VWS_Std_7'] = data['Volume-Weighted Sentiment'].rolling(window=7).std()

    # Calculate Normalized VWS (7 Days) using Z-Score normalization
    data['Normalized VWS (7 Days)'] = (
        data['Volume-Weighted Sentiment'] - data['VWS_Mean_7']
    ) / data['VWS_Std_7']

    # Fill NaN values with 0 for the first 6 rows where rolling calculations are incomplete
    data['Normalized VWS (7 Days)'].fillna(0, inplace=True)

    # Drop intermediate rolling columns
    data.drop(['VWS_Mean_7', 'VWS_Std_7'], axis=1, inplace=True)

    # Debug: Check the recalculated Normalized VWS (7 Days)
    print(f"Recalculated Normalized VWS (7 Days) for {ticker_folder}:")
    print(data[['Date', 'Volume-Weighted Sentiment', 'Normalized VWS (7 Days)']].head(10))

    # Save the prepared data for the ticker in the same folder as the input file
    output_file_path = os.path.join(ticker_path, f"{ticker_folder}_prepared_data.csv")
    data.to_csv(output_file_path, index=False)

    print(f"Data preparation complete for {ticker_folder}. Prepared data saved to {output_file_path}.")
