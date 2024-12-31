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

# # Step 2: Graph Construction for GNNs
# def construct_graph_with_features(tickers, data_dir):
#     G = nx.Graph()
#     for ticker in tickers:
#         file_path = f"{data_dir}/{ticker}/{ticker}_features.csv"
#         if not os.path.exists(file_path):
#             print(f"Features not found for {ticker}, skipping...")
#             continue

#         df = pd.read_csv(file_path)
#         for _, row in df.iterrows():
#             # Add nodes with attributes
#             G.add_node(
#                 row['Ticker Name'],
#                 date=row['Date'],
#                 sector=row['Sector'],
#                 industry=row['Industry'],
#                 volume_weighted_sentiment=row['Volume-Weighted Sentiment'],
#                 normalized_vws_7=row[f'Normalized VWS ({rolling_window} Days)'],
#                 daily_return=row['Daily Return'],
#                 rolling_avg_7=row[f'Rolling Avg ({rolling_window} Days)'],
#                 volatility_7=row[f'Volatility ({rolling_window} Days)'],
#                 momentum_7=row[f'Momentum ({rolling_window} Days)']
#             )
    
#     print(f"Graph with features constructed for {len(G.nodes)} nodes.")
#     return G

# # Save graph to file
# def save_graph_to_file(graph, output_file):
#     nx.write_gml(graph, output_file)
#     print(f"Graph saved to {output_file}")

# # Step 3: Data Preparation for Transformer Models
# def prepare_transformer_input(tickers, data_dir):
#     transformer_data = []
#     for ticker in tickers:
#         file_path = f"{data_dir}/{ticker}/{ticker}_features.csv"
#         if not os.path.exists(file_path):
#             print(f"Features not found for {ticker}, skipping...")
#             continue

#         df = pd.read_csv(file_path)
#         for _, row in df.iterrows():
#             transformer_data.append({
#                 "ticker": row['Ticker Name'],
#                 "date": row['Date'],
#                 "volume_weighted_sentiment": row['Volume-Weighted Sentiment'],
#                 "normalized_vws_7": row[f'Normalized VWS ({rolling_window} Days)'],
#                 "daily_return": row['Daily Return'],
#                 "rolling_avg_7": row[f'Rolling Avg ({rolling_window} Days)'],
#                 "volatility_7": row[f'Volatility ({rolling_window} Days)'],
#                 "momentum_7": row[f'Momentum ({rolling_window} Days)']
#             })
#     print(f"Transformer input prepared for {len(transformer_data)} records.")
#     return transformer_data

# # Save transformer data to file
# def save_transformer_data(data, output_file):
#     pd.DataFrame(data).to_csv(output_file, index=False)
#     print(f"Transformer data saved to {output_file}")

# # Execute for each ticker
# for ticker in tickers:
#     calculate_features(ticker)

# # Construct graph and save
# graph = construct_graph_with_features(tickers, base_path)
# save_graph_to_file(graph, "stock_graph_with_features.gml")

# # Prepare transformer input and save
# transformer_input = prepare_transformer_input(tickers, base_path)
# save_transformer_data(transformer_input, "transformer_input.csv")
