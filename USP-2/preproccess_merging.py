'''
import pandas as pd
import os

tickers = [
    'PD', 'HEAR', 'AAPL', 'PANW', 'ARRY', 'TEL', 'ARQQ', 'ANET', 'UI', 'ZM', 'AGYS', 'FSLR',
    'INOD', 'UBER', 'SNPS', 'ADI', 'FORM', 'PLTR', 'SQ', 'RELL', 'AMKR', 'CSIQ', 'KD', 'BL',
    'TXN', 'KLAC', 'INTC', 'APP', 'BILL', 'RELY', 'CDNS', 'MU', 'APLD', 'FIS', 'TOST', 'DDOG',
    'AMAT', 'PAYO', 'NTAP', 'FICO', 'TTD', 'ATOM', 'DMRC', 'ENPH', 'TWLO', 'BMI', 'BMBL',
    'MSTR', 'OLED', 'CRWD', 'SOUN', 'LYFT', 'PATH', 'QCOM', 'BAND', 'RUM', 'ESTC', 'DOCU',
    'U', 'SEDG', 'MSFT', 'HPE', 'COHR', 'MEI', 'ONTO', 'ACN', 'WK', 'HPQ', 'S', 'GEN', 'ORCL',
    'OUST', 'DELL', 'ALAB', 'ODD', 'IBM', 'ADP', 'AMD', 'WOLF', 'LRCX', 'PI', 'SMCI', 'PAGS',
    'STNE', 'NXT', 'ZS', 'WDAY', 'HUBS', 'BTDR', 'NOW', 'AI', 'AFRM', 'NTNX', 'CORZ', 'KN',
    'INTU', 'WDC', 'TASK', 'ACMR', 'APH', 'OKTA', 'NEON', 'DOX', 'AEHR', 'CSCO', 'NVMI',
    'CRSR', 'SMTC', 'KEYS', 'AVGO', 'OS', 'RAMP', 'NCNO', 'INSG', 'KOSS', 'AAOI', 'SNOW',
    'ADSK', 'PSFE', 'RUN', 'UIS', 'ASTS', 'ON', 'KPLT', 'ADBE', 'FLEX', 'CAMT', 'QXO', 'AIP',
    'VSAT', 'BASE', 'MAXN', 'PGY', 'TDY', 'PAR', 'MRVL', 'PLUS', 'GCT', 'BKSY',
    'FOUR'
]

base_path = '/Users/abhishekjoshi/Documents/GitHub/stock_forecasting_CAI/data'

def process_sentiment_data(sentiment_df):
    # Group by Date and aggregate the data
    grouped_df = sentiment_df.groupby('Date').agg(
        Cumulative_Score=('Cumulative Score', 'sum'),
        Confidence=('Confidence', lambda x: (x * sentiment_df.loc[x.index, 'Total Total text Count']).sum() / sentiment_df.loc[x.index, 'Total Total text Count'].sum()),
        Text_Count=('Total text Count', 'sum'),
    ).reset_index()

    # Calculate Normalized Score
    grouped_df['Normalized_Score'] = grouped_df['Cumulative_Score'] / grouped_df['Text_Count']
    grouped_df['Normalized_Score'] = grouped_df['Normalized_Score'].fillna(0)  # Handle division by zero

    return grouped_df

def merge_ticker_data(ticker):
    try:
        sentiment_file = os.path.join(base_path, f"{ticker}/{ticker}_daily_scores.csv")
        historic_file = os.path.join(base_path, f"{ticker}/{ticker}_historic_data.csv")
        
        if not os.path.exists(sentiment_file) or not os.path.exists(historic_file):
            print(f"Files missing for {ticker}, skipping...")
            return
        
        sentiment_df = pd.read_csv(sentiment_file)
        historic_df = pd.read_csv(historic_file)

        # Process sentiment data
        grouped_sentiment_df = process_sentiment_data(sentiment_df)

        # Rename columns for merging
        grouped_sentiment_df.rename(columns={
            'Cumulative_Score': 'Sentiment Score',
            'Normalized_Score': 'Normalized Score',
            'Text_Count': 'Total text Count'
        }, inplace=True)

        # Merge with historical data
        merged_df = pd.merge(
            historic_df,
            grouped_sentiment_df,
            on='Date',
            how='left'
        )
        
        # Save the merged data
        output_file = os.path.join(base_path, f"{ticker}/{ticker}_historic_data_updated.csv")
        merged_df.to_csv(output_file, index=False)
        print(f"Processed {ticker}: Updated file saved to {output_file}")
    
    except Exception as e:
        print(f"Error processing {ticker}: {e}")

for ticker in tickers:
    merge_ticker_data(ticker)
'''

#Volume Weighted Sentimental
import pandas as pd
import os

tickers = [
    'PD', 'HEAR', 'AAPL', 'PANW', 'ARRY', 'TEL', 'ARQQ', 'ANET', 'UI', 'ZM', 'AGYS', 'FSLR',
    'INOD', 'UBER', 'SNPS', 'ADI', 'FORM', 'PLTR', 'SQ', 'RELL', 'AMKR', 'CSIQ', 'KD', 'BL',
    'TXN', 'KLAC', 'INTC', 'APP', 'BILL', 'RELY', 'CDNS', 'MU', 'APLD', 'FIS', 'TOST', 'DDOG',
    'AMAT', 'PAYO', 'NTAP', 'FICO', 'TTD', 'ATOM', 'DMRC', 'ENPH', 'TWLO', 'BMI', 'BMBL',
    'MSTR', 'OLED', 'CRWD', 'SOUN', 'LYFT', 'PATH', 'QCOM', 'BAND', 'RUM', 'ESTC', 'DOCU',
    'U', 'SEDG', 'MSFT', 'HPE', 'COHR', 'MEI', 'ONTO', 'ACN', 'WK', 'HPQ', 'S', 'GEN', 'ORCL',
    'OUST', 'DELL', 'ALAB', 'ODD', 'IBM', 'ADP', 'AMD', 'WOLF', 'LRCX', 'PI', 'SMCI', 'PAGS',
    'STNE', 'NXT', 'ZS', 'WDAY', 'HUBS', 'BTDR', 'NOW', 'AI', 'AFRM', 'NTNX', 'CORZ', 'KN',
    'INTU', 'WDC', 'TASK', 'ACMR', 'APH', 'OKTA', 'NEON', 'DOX', 'AEHR', 'CSCO', 'NVMI',
    'CRSR', 'SMTC', 'KEYS', 'AVGO', 'OS', 'RAMP', 'NCNO', 'INSG', 'KOSS', 'AAOI', 'SNOW',
    'ADSK', 'PSFE', 'RUN', 'UIS', 'ASTS', 'ON', 'KPLT', 'ADBE', 'FLEX', 'CAMT', 'QXO', 'AIP',
    'VSAT', 'BASE', 'MAXN', 'NVDA', 'PGY', 'TDY', 'PAR', 'MRVL', 'PLUS', 'GCT', 'BKSY',
    'FOUR'
]

base_path = '/Users/abhishekjoshi/Documents/GitHub/stock_forecasting_CAI/data'

# Function to process and group sentiment data
def process_sentiment_data(sentiment_df):
    # Group by Date and aggregate the data
    grouped_df = sentiment_df.groupby('Date').agg(
        Cumulative_Score=('Cumulative Score', 'sum'),
        Confidence=('Confidence', lambda x: (x * sentiment_df.loc[x.index, 'Total Text Count']).sum() / sentiment_df.loc[x.index, 'Total Text Count'].sum()),
        Text_Count=('Total Text Count', 'sum')
    ).reset_index()

    # Calculate Normalized Score
    grouped_df['Normalized_Score'] = grouped_df['Cumulative_Score'] / grouped_df['Text_Count']
    grouped_df['Normalized_Score'] = grouped_df['Normalized_Score'].fillna(0)  # Handle division by zero

    return grouped_df

# Function to merge sentiment data with historical data
def merge_ticker_data(ticker):
    try:
        sentiment_file = os.path.join(base_path, f"{ticker}/{ticker}_daily_scores.csv")
        historic_file = os.path.join(base_path, f"{ticker}/{ticker}_historic_data.csv")
        
        if not os.path.exists(sentiment_file) or not os.path.exists(historic_file):
            print(f"Files missing for {ticker}, skipping...")
            return
        
        sentiment_df = pd.read_csv(sentiment_file)
        historic_df = pd.read_csv(historic_file)

        # Process sentiment data
        grouped_sentiment_df = process_sentiment_data(sentiment_df)

        # Rename columns for merging
        grouped_sentiment_df.rename(columns={
            'Cumulative_Score': 'Sentiment Score',
            'Normalized_Score': 'Normalized Score',
            'Text_Count': 'Total Text Count'
        }, inplace=True)

        # Merge with historical data
        merged_df = pd.merge(
            historic_df,
            grouped_sentiment_df,
            on='Date',
            how='left'
        )
        
        # Fill missing values
        merged_df[['Sentiment Score', 'Confidence', 'Total Text Count', 'Normalized Score']] = merged_df[
            ['Sentiment Score', 'Confidence', 'Total Text Count', 'Normalized Score']
        ].fillna(0)

        # Save the merged data
        output_file = os.path.join(base_path, f"{ticker}/{ticker}_historic_data_updated.csv")
        merged_df.to_csv(output_file, index=False)
        print(f"Processed {ticker}: Updated file saved to {output_file}")
    
    except Exception as e:
        print(f"Error processing {ticker}: {e}")

# Function to calculate Volume-Weighted Sentiment and normalize it
def calculate_volume_weighted_sentiment(ticker):
    try:
        input_file = os.path.join(base_path, f"{ticker}/{ticker}_historic_data_updated.csv")
        
        if not os.path.exists(input_file):
            print(f"Updated file missing for {ticker}, skipping...")
            return
        
        # Load the merged data
        data = pd.read_csv(input_file)

        # Ensure necessary columns are present
        required_columns = ['Date', 'Sentiment Score', 'Volume', 'Confidence', 'Total Text Count', 'Normalized Score']
        for col in required_columns:
            if col not in data.columns:
                print(f"Missing required column {col} in {ticker}, skipping...")
                return
        
        # Fill missing values with defaults
        data['Sentiment Score'] = data['Sentiment Score'].fillna(0)
        data['Confidence'] = data['Confidence'].fillna(0)
        data['Total Text Count'] = data['Total Text Count'].fillna(0)
        data['Normalized Score'] = data['Normalized Score'].fillna(0)
        data['Volume'] = data['Volume'].fillna(1)  # Avoid division by zero

        # Calculate Volume-Weighted Sentiment
        data['Volume-Weighted Sentiment'] = data['Sentiment Score'] * data['Volume']

        # Normalize Volume-Weighted Sentiment over a rolling window (e.g., 7 days)
        rolling_window_size = 7
        data['Max_Volume_Weighted_Sentiment'] = data['Volume-Weighted Sentiment'].rolling(
            window=rolling_window_size, min_periods=1
        ).max()

        # Avoid division by zero for normalization
        data['Max_Volume_Weighted_Sentiment'] = data['Max_Volume_Weighted_Sentiment'].replace(0, 1)

        data['Normalized Sentiment'] = (
            data['Volume-Weighted Sentiment'] / data['Max_Volume_Weighted_Sentiment']
        )

        # Drop helper column if not needed
        data.drop(columns=['Max_Volume_Weighted_Sentiment'], inplace=True)

        # Save the updated data
        output_file = os.path.join(base_path, f"{ticker}/{ticker}_historic_data_vws.csv")
        data.to_csv(output_file, index=False)
        print(f"Processed {ticker}: VWS data saved to {output_file}")
    
    except Exception as e:
        print(f"Error processing {ticker}: {e}")

# Process all tickers
for ticker in tickers:
    merge_ticker_data(ticker)
    calculate_volume_weighted_sentiment(ticker)
