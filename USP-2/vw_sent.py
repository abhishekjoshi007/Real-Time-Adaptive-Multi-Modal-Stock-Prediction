import pandas as pd
import os

def calculate_volume_weighted_sentiment(ticker):
    # Define the path to the updated historic data file
    updated_data_path = f"/Users/gurojaschadha/Downloads/data 2/{ticker}/{ticker}_historic_data_updated.csv"

    # Check if the file exists
    if not os.path.exists(updated_data_path):
        print(f"File not found for {ticker}: {updated_data_path}")
        return

    # Load the updated historic data
    stock_df = pd.read_csv(updated_data_path)

    # Ensure necessary columns exist before proceeding
    if 'Z_Volume' not in stock_df.columns or 'Sentiment Score' not in stock_df.columns:
        print(f"Necessary columns missing in {updated_data_path}")
        return

    # 2.1 Compute Volume-Weighted Sentiment (VW Sentiment)
    stock_df['VW_Sentiment'] = stock_df['Z_Volume'] * stock_df['Sentiment Score']

    # 2.2 Calculate EMA for Volume-Weighted Sentiment (VW Sentiment)
    stock_df['VW_Sentiment_EMA'] = stock_df['VW_Sentiment'].ewm(span=20, adjust=False).mean()

    # 2.3 Calculate EMA for the standard deviation of Volume-Weighted Sentiment
    stock_df['VW_Sentiment_EMA_Std'] = stock_df['VW_Sentiment'].ewm(span=20, adjust=False).std()

    # 2.4 Normalize Volume-Weighted Sentiment using EMA
    stock_df['Normalized_VW_Sentiment'] = (stock_df['VW_Sentiment'] - stock_df['VW_Sentiment_EMA']) / stock_df['VW_Sentiment_EMA_Std']

    # Reorder columns to add the new VW Sentiment columns before sentiment-related columns
    column_order = [
        'Date', 'Ticker Name', 'Sector', 'Industry', 'Market Cap', 'Open', 'High', 'Low', 'Close', 'Adj Close',
        'Volume', 'Z_Volume', 'EMA_Volume', 'EMA_Volume_Std', 'Sentiment Score', 'Confidence', 'Normalized Score',
        'VW_Sentiment', 'VW_Sentiment_EMA', 'VW_Sentiment_EMA_Std', 'Normalized_VW_Sentiment'
    ]
    stock_df = stock_df[column_order]

    # Save the updated data back to the same file
    stock_df.to_csv(updated_data_path, index=False)
    print(f"Volume-Weighted Sentiment calculations (with EMA) saved for {ticker} at {updated_data_path}")

# List of tickers
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

# Process each ticker's updated historic data
for ticker in tickers:
    calculate_volume_weighted_sentiment(ticker)
