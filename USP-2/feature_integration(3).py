#Adding volume weigthed Sentiments
import pandas as pd
import os
import networkx as nx

# Define tickers and data directory
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

# Rolling window size
rolling_window = 7

# Step 1: Feature Calculation
def calculate_features(ticker):
    input_file = os.path.join(base_path, f"{ticker}/{ticker}_historic_data_vws.csv")
    output_file = os.path.join(base_path, f"{ticker}/{ticker}_features.csv")
    
    if not os.path.exists(input_file):
        print(f"Updated file missing for {ticker}, skipping...")
        return
    
    data = pd.read_csv(input_file)
    
    # Ensure necessary columns exist
    required_columns = ['Date', 'Close', 'Volume', 'Sentiment Score', 'Volume-Weighted Sentiment', 'Normalized Sentiment']
    for col in required_columns:
        if col not in data.columns:
            print(f"Missing required column {col} in {ticker}, skipping...")
            return

    # Rename Normalized Sentiment to Normalized VWS (7 Days)
    data.rename(columns={'Normalized Sentiment': f'Normalized VWS ({rolling_window} Days)'}, inplace=True)

    # Fill missing values
    data['Close'] = data['Close'].fillna(method='ffill')
    data['Volume'] = data['Volume'].fillna(0)
    data['Sentiment Score'] = data['Sentiment Score'].fillna(0)
    data['Volume-Weighted Sentiment'] = data['Volume-Weighted Sentiment'].fillna(0)

    # Calculate daily returns
    data['Daily Return'] = data['Close'].pct_change().fillna(0)

    # Initialize rolling features
    data[f'Rolling Avg ({rolling_window} Days)'] = data['Close']
    data[f'Volatility ({rolling_window} Days)'] = data['Close']
    data[f'Momentum ({rolling_window} Days)'] = data['Close']

    # Calculate rolling features for rows starting from the 2nd entry
    data.loc[1:, f'Rolling Avg ({rolling_window} Days)'] = (
        data['Close'].rolling(window=rolling_window, min_periods=1).mean()
    )
    data.loc[1:, f'Volatility ({rolling_window} Days)'] = (
        data['Close'].rolling(window=rolling_window, min_periods=1).std()
    )

    # Calculate momentum, set first 7 days as Close value
    data[f'Momentum ({rolling_window} Days)'] = data['Close'] - data['Close'].shift(rolling_window)
    data.loc[:rolling_window-1, f'Momentum ({rolling_window} Days)'] = data.loc[:rolling_window-1, 'Close']

    # Save the features to a file
    data.to_csv(output_file, index=False)
    print(f"Features calculated for {ticker}: {output_file}")

    # Execute for each ticker
for ticker in tickers:
    calculate_features(ticker)
