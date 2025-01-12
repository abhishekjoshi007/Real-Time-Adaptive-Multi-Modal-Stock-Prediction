import os  # For directory management
import pandas as pd
import numpy as np  # For numerical operations
from sklearn.preprocessing import MinMaxScaler

# Define the root data directory
root_data_dir = "/Users/abhishekjoshi/Documents/GitHub/stock_forecasting_CAI/Data 2"

# Loop through each ticker folder
for ticker_folder in os.listdir(root_data_dir):
    ticker_path = os.path.join(root_data_dir, ticker_folder)
    
    # Check if it's a directory (ignore files or unrelated content)
    if os.path.isdir(ticker_path):
        input_file = os.path.join(ticker_path, f"{ticker_folder}_merged_with_vix.csv")
        
        # Check if the input file exists
        if os.path.exists(input_file):
            try:
                # Load data
                data = pd.read_csv(input_file)
                
                # Define relevant columns
                selected_columns = ['Volume-Weighted Sentiment', 'Event_Flag', 'Volatility (7 Days)', 'EWMA Volatility']
                missing_columns = [col for col in selected_columns if col not in data.columns]
                
                # Skip if any required column is missing
                if missing_columns:
                    print(f"Skipping {ticker_folder}: Missing columns {missing_columns}")
                    continue
                
                # Select relevant columns
                features = data[selected_columns]
                
                # Normalize using Min-Max Scaling
                scaler = MinMaxScaler()
                normalized_features = scaler.fit_transform(features[['Volume-Weighted Sentiment', 'Volatility (7 Days)', 'EWMA Volatility']])
                
                # Add Event_Flag as binary (already in 0/1 format)
                event_flag = features['Event_Flag'].values.reshape(-1, 1)
                
                # Combine normalized features and Event_Flag
                normalized_tensor = np.hstack((normalized_features, event_flag))
                input_tensor = pd.DataFrame(
                    data=normalized_tensor,
                    columns=['Normalized VWS', 'Normalized Volatility', 'Normalized EWMA Volatility', 'Event Flag']
                )
                
                # Add Ticker Name, Date, and Close Price
                if 'Ticker Name' in data.columns and 'Date' in data.columns and 'Close' in data.columns:
                    input_tensor['Ticker Name'] = data['Ticker Name']
                    input_tensor['Date'] = data['Date']
                    input_tensor['Close Price'] = data['Close']
                else:
                    print(f"Skipping {ticker_folder}: Missing 'Ticker Name', 'Date', or 'Close' column")
                    continue
                
                # Reorder columns for better readability
                input_tensor = input_tensor[['Ticker Name', 'Date', 'Close Price', 'Normalized VWS', 'Normalized Volatility', 'Normalized EWMA Volatility', 'Event Flag']]
                
                # Save prepared data to the respective ticker folder
                output_file = os.path.join(ticker_path, f"{ticker_folder}_usp3_prepared_data.csv")
                input_tensor.to_csv(output_file, index=False)
                
                print(f"Processed and saved data for ticker: {ticker_folder} to {output_file}")
            except Exception as e:
                print(f"Error processing ticker {ticker_folder}: {e}")
        else:
            print(f"Input file not found for ticker: {ticker_folder}")
