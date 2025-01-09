import os

def remove_ticker_files(base_dir):
    """
    Removes specific files for each ticker folder in the given base directory.

    Args:
        base_dir (str): The path to the base directory containing ticker subfolders.
    """
    # File patterns to be removed
    file_patterns = [
        "{ticker}_comments.json",
        "{ticker}_daily_scores.csv",
        "{ticker}_detailed_scores.csv",
        "{ticker}_historic_data_updated.csv",
        "{ticker}_historic_data_vws.csv",
        "{ticker}_historic_data.csv",
        "{ticker}_holder.json",
        "evaluation_results_{ticker}.json",
        "{ticker}_merged.csv",
        "{ticker}_USP1_features.csv",
        "{ticker}_filtered_stocks.csv",
        "{ticker}_final.csv",
         "{ticker}_prepared_data.csv",
          "{ticker}_processed.csv",
           "{ticker}_features.csv",
            "{ticker}_refined.csv",
           

    ]

    # Iterate through subfolders
    for ticker_folder in os.listdir(base_dir):
        ticker_path = os.path.join(base_dir, ticker_folder)

        if os.path.isdir(ticker_path):
            for file_pattern in file_patterns:
                file_to_remove = os.path.join(ticker_path, file_pattern.format(ticker=ticker_folder))

                # Remove the file if it exists
                if os.path.exists(file_to_remove):
                    os.remove(file_to_remove)
                    print(f"Removed: {file_to_remove}")
                else:
                    print(f"File not found: {file_to_remove}")

# Example usage
base_directory = "/Users/abhishekjoshi/Documents/GitHub/stock_forecasting_CAI/Data 2"
remove_ticker_files(base_directory)
