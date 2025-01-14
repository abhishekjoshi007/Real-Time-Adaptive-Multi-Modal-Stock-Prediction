import os
import pandas as pd

# Define paths for USP 1+2 and USP 3 data folders
usp1_2_base_path = "/Users/abhishekjoshi/Documents/GitHub/stock_forecasting_CAI/USP 1 Data"  # Replace with your actual data1 folder path
usp3_base_path = "/Users/abhishekjoshi/Documents/GitHub/stock_forecasting_CAI/USP 3 data"    # Replace with your actual data2 folder path
output_base_path = "merged_data_usp1_usp3"  # Folder to save merged files (create this directory if it doesn't exist)

# Ensure the output directory exists
os.makedirs(output_base_path, exist_ok=True)

# Iterate over all ticker folders in data1
for ticker_folder in os.listdir(usp1_2_base_path):
    usp1_2_folder_path = os.path.join(usp1_2_base_path, ticker_folder)
    usp3_folder_path = os.path.join(usp3_base_path, ticker_folder)
    
    # Check if both folders exist and contain the respective files
    usp1_2_file = os.path.join(usp1_2_folder_path, f"{ticker_folder}_merged_with_vix.csv")
    usp3_file = os.path.join(usp3_folder_path, f"{ticker_folder}_usp3_prepared_data.csv")
    
    if os.path.exists(usp1_2_file) and os.path.exists(usp3_file):
        # Read the CSV files
        usp1_2_df = pd.read_csv(usp1_2_file)
        usp3_df = pd.read_csv(usp3_file)
        
        # Merge the DataFrames on common keys ('Date' and 'Ticker Name')
        merged_df = pd.merge(
            usp1_2_df, usp3_df, on=['Date', 'Ticker Name'], how='outer', suffixes=('_USP1_2', '_USP3')
        )
        
        # Handle duplicate columns (example for overlapping features like 'Close')
        duplicate_columns = ['Close']  # Add other overlapping column names as needed
        for col in duplicate_columns:
            if f'{col}_USP1_2' in merged_df.columns and f'{col}_USP3' in merged_df.columns:
                merged_df[col] = merged_df[f'{col}_USP3']  # Prioritize USP 3 data
                merged_df.drop([f'{col}_USP1_2', f'{col}_USP3'], axis=1, inplace=True)
        
        # Drop duplicate rows across all columns
        merged_df.drop_duplicates(inplace=True)
        
        # Save the merged DataFrame to the output directory
        output_file = os.path.join(output_base_path, f"{ticker_folder}.csv")
        merged_df.to_csv(output_file, index=False)
        print(f"Successfully merged and saved data for ticker: {ticker_folder}")
    else:
        print(f"Missing files for ticker: {ticker_folder}. Skipping.")
