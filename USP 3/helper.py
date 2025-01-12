import os
import pandas as pd

def merge_csvs(csv1_path, csv2_path, output_path):
    """
    Merges two CSV files into a single file with the required columns.
    
    Args:
        csv1_path (str): Path to the first CSV file (historical data).
        csv2_path (str): Path to the second CSV file (additional data).
        output_path (str): Path to save the merged CSV file.
    """
    # Load both CSV files into DataFrames
    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)

    # Select relevant columns from each DataFrame
    df1 = df1[['Ticker Name', 'Date', 'Normalized VWS', 
               'Normalized Volatility', 'Normalized EWMA Volatility', 'Event Flag']]
    df2 = df2[['Date', 'Ticker Name', 'Close', 'Volatility (7 Days)']]

    # Merge the DataFrames on 'Date' and 'Ticker Name'
    merged_df = pd.merge(df1, df2, on=['Date', 'Ticker Name'], how='inner')

    # Save the merged DataFrame to the output path
    merged_df.to_csv(output_path, index=False)
    print(f"Merged CSV saved to {output_path}")


def process_tickers(base_dir):
    """
    Processes all tickers in the specified base directory and merges their CSV files.
    
    Args:
        base_dir (str): Path to the base directory containing ticker folders.
    """
    for ticker_folder in os.listdir(base_dir):
        ticker_path = os.path.join(base_dir, ticker_folder)
        if os.path.isdir(ticker_path):
            # Define paths for the two input CSVs and the output CSV
            csv1_path = os.path.join(ticker_path, f"{ticker_folder}_usp3_prepared_data.csv")
            csv2_path = os.path.join(ticker_path, f"{ticker_folder}_merged_with_vix.csv")
            output_path = os.path.join(ticker_path, f"{ticker_folder}_usp3_prepared_data.csv")

            # Check if both input CSV files exist
            if os.path.exists(csv1_path) and os.path.exists(csv2_path):
                try:
                    # Call the merge function
                    merge_csvs(csv1_path, csv2_path, output_path)
                except Exception as e:
                    print(f"Error processing {ticker_folder}: {e}")
            else:
                print(f"Missing files for ticker {ticker_folder}. Skipping...")


if __name__ == "__main__":
    # Base directory containing ticker folders
    base_dir = "/Users/abhishekjoshi/Documents/GitHub/stock_forecasting_CAI/Data 2"  # Replace with your base directory

    # Process all tickers in the base directory
    process_tickers(base_dir)
