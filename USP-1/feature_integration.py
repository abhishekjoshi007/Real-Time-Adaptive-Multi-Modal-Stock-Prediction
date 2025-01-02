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
base_path = '/Users/abhishekjoshi/Documents/GitHub/stock_forecasting_CAI/data'

# Rolling window size
rolling_window = 7

# Step 1: Feature Calculation
def calculate_features(ticker):
    input_file = os.path.join(base_path, f"{ticker}/{ticker}_historic_data_vws.csv")
    output_file = os.path.join(base_path, f"{ticker}/{ticker}_features.csv")
    
    if not os.path.exists(input_file):
        print(f"Input file missing for {ticker}, skipping...")
        return
    
    data = pd.read_csv(input_file)
    
    # Ensure necessary columns exist
    required_columns = ['Date', 'Close']
    for col in required_columns:
        if col not in data.columns:
            print(f"Missing required column {col} in {ticker}, skipping...")
            return

    # Fill missing values for Close
    data['Close'] = data['Close'].fillna(method='ffill')

    # Calculate daily returns
    data['Daily Return'] = data['Close'].pct_change().fillna(0)

    # Calculate rolling average and rolling volatility
    data[f'Rolling Avg ({rolling_window} Days)'] = (
        data['Close'].rolling(window=rolling_window, min_periods=1).mean()
    )
    data[f'Volatility ({rolling_window} Days)'] = (
        data['Close'].rolling(window=rolling_window, min_periods=1).std()
    )

    # Calculate Exponentially Weighted Moving Average (EWMA) volatility
    ewma_span = rolling_window  # You can adjust the span for smoother or more reactive EWMA
    data['EWMA Volatility'] = data['Daily Return'].ewm(span=ewma_span, adjust=False).std()

    # Step 2: Volatility Thresholds
    ewma_mean = data['EWMA Volatility'].mean()
    ewma_std = data['EWMA Volatility'].std()
    k = 1  # Adjust sensitivity here (1 to 2)

    # Define thresholds
    high_threshold = ewma_mean + k * ewma_std
    low_threshold = ewma_mean - k * ewma_std

    # Assign volatility class
    def classify_volatility(volatility):
        if volatility > high_threshold:
            return 'High'
        elif volatility < low_threshold:
            return 'Low'
        else:
            return 'Medium'

    data['Volatility Class'] = data['EWMA Volatility'].apply(classify_volatility)

    # Save the features to a file
    data.to_csv(output_file, index=False)
    print(f"Features calculated for {ticker}: {output_file}")

# Execute for each ticker
for ticker in tickers:
    calculate_features(ticker)
