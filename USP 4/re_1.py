import pandas as pd
import os

# Define input and output directories
input_dir = '/Users/abhishekjoshi/Documents/GitHub/stock_forecasting_CAI/Data 2'  # Directory where rank score files for all tickers are stored
output_file = 'daily_recommendations.csv'  # Path to save the compiled output

# Initialize a list to store data from all tickers
all_tickers_data = []

# Loop over all ticker folders
for ticker_folder in os.listdir(input_dir):
    ticker_folder_path = os.path.join(input_dir, ticker_folder)
    if os.path.isdir(ticker_folder_path):  # Process only directories
        for file_name in os.listdir(ticker_folder_path):
            if file_name.endswith('_refined.csv'):  # Process only files with ranked data
                file_path = os.path.join(ticker_folder_path, file_name)
                
                # Load the ranked data for the ticker
                df = pd.read_csv(file_path)
                
                # Add the data to the list
                all_tickers_data.append(df)

# Combine data from all tickers
compiled_data = pd.concat(all_tickers_data, ignore_index=True)

# Generate recommendations for each day
recommendations = (
    compiled_data.sort_values(['Date', 'Rank_Score'], ascending=[True, False])  # Sort by Date and Rank_Score
    .groupby('Date')  # Group by Date
    .head(10)  # Select top 10 stocks for each day
)

# Save the recommendations to a single output file
recommendations.to_csv(output_file, index=False)

print(f"Daily recommendations have been compiled and saved to {output_file}")
