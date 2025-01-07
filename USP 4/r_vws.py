import pandas as pd
import os

# Define input and output directories
input_dir = '/Users/abhishekjoshi/Documents/GitHub/stock_forecasting_CAI/Data 2'  # Replace with your base data folder path
output_dir = '/Users/abhishekjoshi/Documents/GitHub/stock_forecasting_CAI/Data 2'  # Replace with your output folder path

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define weights for Rank Score
alpha = 0.7  # Weight for adjusted momentum
beta = 0.3   # Weight for volatility

# Loop over each ticker folder
for ticker_folder in os.listdir(input_dir):
    ticker_folder_path = os.path.join(input_dir, ticker_folder)
    if os.path.isdir(ticker_folder_path):  # Process only directories
        # Paths to input files
        merged_with_vix_file = os.path.join(ticker_folder_path, f"{ticker_folder}_merged_with_vix.csv")
        filtered_stocks_file = os.path.join(ticker_folder_path, f"{ticker_folder}_filtered_stocks.csv")
        
        # Check if both files exist
        if os.path.exists(merged_with_vix_file) and os.path.exists(filtered_stocks_file):
            # Load the data
            merged_with_vix_df = pd.read_csv(merged_with_vix_file)
            filtered_stocks_df = pd.read_csv(filtered_stocks_file)
            
            # Merge Sentiment Score into filtered stocks data
            refined_df = filtered_stocks_df.merge(
                merged_with_vix_df[['Date', 'Sentiment Score']],
                on='Date',
                how='left'
            )
            
            # Handle missing Sentiment Score
            refined_df['Sentiment Score'].fillna(0, inplace=True)
            
            # Adjust Momentum with Sentiment Score
            refined_df['Adjusted_Momentum'] = refined_df['Momentum'] * (1 + refined_df['Sentiment Score'])
            
            # Calculate Rank Score
            refined_df['Rank_Score'] = alpha * refined_df['Adjusted_Momentum'] + beta * refined_df['Volatility']
            
            # Rank based on Rank Score
            refined_df['Rank_Score_Rank'] = refined_df['Rank_Score'].rank(ascending=False, method='first')
            
            # Save the combined output in the respective ticker folder
            ticker_output_dir = os.path.join(output_dir, ticker_folder)
            if not os.path.exists(ticker_output_dir):
                os.makedirs(ticker_output_dir)
            
            output_path = os.path.join(ticker_output_dir, f"{ticker_folder}_refined.csv")
            refined_df.to_csv(output_path, index=False)
            
            print(f"Processed and saved refined data for {ticker_folder} at {output_path}")
