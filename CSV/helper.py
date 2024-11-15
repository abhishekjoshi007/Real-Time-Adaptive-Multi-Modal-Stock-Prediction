import pandas as pd

def merge_csvs(csv1_path, csv2_path, output_path):
    """
    Merges two CSV files based on the 'Ticker' column, retaining all rows from csv1.
    
    Parameters:
    csv1_path (str): Path to the main CSV file.
    csv2_path (str): Path to the secondary CSV file.
    output_path (str): Path where the merged CSV will be saved.
    """
    # Load the CSVs into dataframes
    csv1 = pd.read_csv(csv1_path)
    csv2 = pd.read_csv(csv2_path)

    # Merge the two dataframes on 'Ticker'
    merged_df = pd.merge(csv1, csv2, on="Ticker", how="left")

    # Save the merged dataframe to a new CSV file
    merged_df.to_csv(output_path, index=False)
    print(f"Merged data saved to {output_path}")

# File paths
csv1_path = "merged.csv"  # Replace with your main CSV file path
csv2_path = "merged_data.csv"  # Replace with your secondary CSV file path
output_path = "final_output.csv"  # Replace with desired output file name

# Run the merge function
merge_csvs(csv1_path, csv2_path, output_path)
