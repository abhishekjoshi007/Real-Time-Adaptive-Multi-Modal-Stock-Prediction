import pandas as pd
import os

# Define the base folder path where ticker folders are stored
base_folder = '/Users/abhishekjoshi/Documents/GitHub/stock_forecasting_CAI/Data 2'

# List all tickers (subfolders in the base folder)
tickers = [folder for folder in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, folder))]

# Loop over each ticker
for ticker in tickers:
    try:
        # Define file paths
        input_file = os.path.join(base_folder, ticker, f"{ticker}_final.csv")
        output_file = os.path.join(base_folder, ticker, f"{ticker}_processed.csv")
        
        # Check if the input file exists
        if not os.path.exists(input_file):
            print(f"File not found for {ticker}: {input_file}")
            continue
        
        # Load the dataset
        df = pd.read_csv(input_file)
        
        # Step 1: Calculate Rolling Average and Momentum
        df['Rolling_Avg_Close'] = df['Close'].rolling(window=7).mean()
        df['Momentum'] = df['Close'] - df['Rolling_Avg_Close']
        
        # Step 2: Calculate Volatility (Rolling Standard Deviation)
        df['Volatility'] = df['Close'].rolling(window=7).std()
        
        # Step 3: Calculate EWMA Volatility
        df['EWMA_Volatility'] = df['Close'].ewm(span=7).std()
        
        # Step 4: Normalize Momentum and Volatility
        df['Normalized_Momentum'] = (df['Momentum'] - df['Momentum'].mean()) / df['Momentum'].std()
        df['Normalized_Volatility'] = (df['Volatility'] - df['Volatility'].mean()) / df['Volatility'].std()
        
        # Step 5: Calculate Composite Score
        momentum_weight = 0.7
        volatility_weight = 0.3
        df['Composite_Score'] = (
            momentum_weight * df['Normalized_Momentum'] +
            volatility_weight * df['Normalized_Volatility']
        )
        
        # Step 6: Rank Metrics
        df['Momentum_Rank'] = df['Momentum'].rank(ascending=False, method='first')
        df['Volatility_Rank'] = df['Volatility'].rank(ascending=False, method='first')
        df['Composite_Rank'] = df['Composite_Score'].rank(ascending=False, method='first')
        
        # Step 7: Prepare Final Output
        output_columns = [
            'Date', 'Ticker Name', 'Close', 'Momentum', 'Volatility', 'EWMA_Volatility',
            'Composite_Score', 'Momentum_Rank', 'Volatility_Rank', 'Composite_Rank'
        ]
        final_output = df[output_columns]
        
        # Save the output to a CSV file in the same ticker folder
        final_output.to_csv(output_file, index=False)
        print(f"Processed {ticker}, saved to {output_file}")
    
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
