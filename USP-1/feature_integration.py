import pandas as pd
import os

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
    'VSAT', 'BASE', 'MAXN', 'PGY', 'TDY', 'PAR', 'MRVL', 'PLUS', 'GCT', 'BKSY',
    'FOUR'
]


base_path = '/Users/abhishekjoshi/Documents/GitHub/stock_forecasting_CAI/Data 2'

# Rolling window size
rolling_window = 7

# Feature Calculation
def calculate_features(ticker):
    input_file = os.path.join(base_path, f"{ticker}/{ticker}_features.csv")
    output_file = os.path.join(base_path, f"{ticker}/{ticker}_USP1_features.csv")
    
    if not os.path.exists(input_file):
        print(f"Input file missing for {ticker}, skipping...")
        return
    
    # Read input data
    data = pd.read_csv(input_file)
    
    # Ensure necessary columns exist
    if 'Close' not in data.columns:
        print(f"Missing 'Close' column for {ticker}, skipping...")
        return

    # Preserve original data
    original_data = data.copy()
    
    # Fill missing values for 'Close' using forward fill
    data['Close'] = data['Close'].fillna(method='ffill')
    
    # Calculate Daily Returns
    data['Daily Return'] = data['Close'].pct_change().fillna(0)
    
    # Calculate EWMA Volatility
    ewma_span = rolling_window
    data['EWMA Volatility'] = data['Daily Return'].ewm(span=ewma_span, adjust=False).std()

    # Handle missing EWMA Volatility for the first day
    if pd.isna(data.loc[0, 'EWMA Volatility']):
        # Use the first rolling window's standard deviation of returns
        first_volatility = data['Daily Return'][:rolling_window].std()
        data.loc[0, 'EWMA Volatility'] = first_volatility

    # Forward fill any remaining missing EWMA Volatility values
    data['EWMA Volatility'] = data['EWMA Volatility'].fillna(method='ffill')

    # Calculate Volatility Class
    ewma_mean = data['EWMA Volatility'].mean()
    ewma_std = data['EWMA Volatility'].std()
    k = 1  # Sensitivity factor
    high_threshold = ewma_mean + k * ewma_std
    low_threshold = ewma_mean - k * ewma_std

    def classify_volatility(volatility):
        if volatility > high_threshold:
            return 'High'
        elif volatility < low_threshold:
            return 'Low'
        else:
            return 'Medium'

    data['Volatility Class'] = data['EWMA Volatility'].apply(classify_volatility)

    # Append only new columns to original data
    original_data['Daily Return'] = data['Daily Return']
    original_data['EWMA Volatility'] = data['EWMA Volatility']
    original_data['Volatility Class'] = data['Volatility Class']

    # Save output
    original_data.to_csv(output_file, index=False)
    print(f"Features calculated for {ticker}: {output_file}")

# Execute for each ticker
for ticker in tickers:
    calculate_features(ticker)
