'''
#merging historic and sentiment data for each ticker
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

base_path = '/Users/gurojaschadha/Downloads/data 2/'

def merge_ticker_data(ticker):
    sentiment_file = os.path.join(base_path, f"{ticker}/{ticker}_daily_scores.csv")
    historic_file = os.path.join(base_path, f"{ticker}/{ticker}_historic_data.csv")
    
    if not os.path.exists(sentiment_file) or not os.path.exists(historic_file):
        print(f"Files missing for {ticker}, skipping...")
        return
    
    sentiment_df = pd.read_csv(sentiment_file)
    historic_df = pd.read_csv(historic_file)
    
    sentiment_df.rename(columns={'Cumulative Score': 'Sentiment Score'}, inplace=True)
    
    merged_df = pd.merge(
        historic_df,
        sentiment_df[['Date', 'Sentiment Score', 'Confidence', 'Normalized Score']],
        on='Date',
        how='left'
    )
    
    output_file = os.path.join(base_path, f"{ticker}/{ticker}_historic_data_updated.csv")
    merged_df.to_csv(output_file, index=False)
    print(f"Processed {ticker}: Updated file saved to {output_file}")

for ticker in tickers:
    merge_ticker_data(ticker)
'''
#preprocessing data
import pandas as pd
import os

def preprocess_data(ticker):
    updated_data_path = f"/Users/gurojaschadha/Downloads/data_vw_sentiment/{ticker}/{ticker}_historic_data_updated.csv"

    if not os.path.exists(updated_data_path):
        print(f"File not found for {ticker}: {updated_data_path}")
        return

    stock_df = pd.read_csv(updated_data_path)

    if 'Volume' not in stock_df.columns:
        print(f"'Volume' column missing in {updated_data_path}")
        return

    # Calculate EMA (Exponential Moving Average) for Volume and its Standard Deviation
    stock_df['EMA_Volume'] = stock_df['Volume'].ewm(span=20, adjust=False).mean()
    stock_df['EMA_Volume_Std'] = stock_df['Volume'].ewm(span=20, adjust=False).std()

    # Calculate Z-Score for Volume
    stock_df['Z_Volume'] = (stock_df['Volume'] - stock_df['EMA_Volume']) / stock_df['EMA_Volume_Std']

    column_order = [
        'Date', 'Ticker Name', 'Sector', 'Industry', 'Market Cap', 'Open', 'High', 'Low', 'Close', 'Adj Close',
        'Volume', 'Z_Volume', 'EMA_Volume', 'EMA_Volume_Std', 'Sentiment Score', 'Confidence', 'Normalized Score'
    ]
    stock_df = stock_df[column_order]

    stock_df.to_csv(updated_data_path, index=False)
    print(f"Updated data with reordered columns saved for {ticker} at {updated_data_path}")

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

for ticker in tickers:
    preprocess_data(ticker)
