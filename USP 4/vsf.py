import pandas as pd
import os
import glob

# Define threshold percentages
momentum_percentile = 80  # Top 20% for Momentum
volatility_percentile = 80  # Top 20% for Volatility

# Function to apply volatility sensitivity filtering on each stock
def volatility_sensitivity_filter(input_folder):
    # Get the list of processed files named like {Ticker_name}_processed.csv
    files = glob.glob(os.path.join(input_folder, "*", "*_processed.csv"))

    for file in files:
        # Read the processed data for each stock
        df = pd.read_csv(file)

        # Calculate thresholds for high momentum and high volatility
        threshold_momentum = df['Momentum'].quantile(momentum_percentile / 100)
        threshold_volatility = df['Volatility'].quantile(volatility_percentile / 100)

        # Filter high-momentum and high-volatility stocks
        filtered_stocks = df[
            (df['Momentum'] > threshold_momentum) & 
            (df['Volatility'] > threshold_volatility)
        ]

        # Save the filtered stocks to the respective ticker's folder
        ticker_folder = os.path.dirname(file)  # Get the folder of the current stock
        ticker_name = os.path.basename(file).split("_")[0]  # Extract ticker name
        output_file = os.path.join(ticker_folder, f"{ticker_name}_filtered_stocks.csv")
        
        filtered_stocks.to_csv(output_file, index=False)

        print(f"Filtered stocks for {ticker_name} saved to {output_file}")

# Specify input folder
input_folder = "/Users/abhishekjoshi/Documents/GitHub/stock_forecasting_CAI/Data 2"  # Replace with your actual folder

# Run the filtering process
volatility_sensitivity_filter(input_folder)
