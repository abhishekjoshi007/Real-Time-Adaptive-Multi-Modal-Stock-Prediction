import pandas as pd
import os
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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

# Thresholds
correlation_threshold = 0.8
similarity_threshold = 0.9
mutual_shareholder_threshold = 5

# Load shareholder data
def load_shareholder_data(ticker):
    file_path = os.path.join(base_path, f"{ticker}/{ticker}_holder.json")
    if os.path.exists(file_path):
        return pd.read_json(file_path)
    return pd.DataFrame()

# Construct graph with edges
def construct_graph_with_edges(tickers, data_dir):
    G = nx.Graph()

    # Load features and shareholders data for all tickers
    feature_data = {}
    shareholder_data = {}
    for ticker in tickers:
        feature_file = os.path.join(data_dir, f"{ticker}/{ticker}_features.csv")
        holder_file = os.path.join(data_dir, f"{ticker}/{ticker}_holder.json")
        
        if not os.path.exists(feature_file):
            print(f"Features not found for {ticker}, skipping...")
            continue
        feature_data[ticker] = pd.read_csv(feature_file)
        
        if os.path.exists(holder_file):
            shareholder_data[ticker] = load_shareholder_data(ticker)

    # Add nodes with features
    for ticker, df in feature_data.items():
        G.add_node(
            ticker,
            sector=df['Sector'][0],
            industry=df['Industry'][0],
            volume_weighted_sentiment=df['Volume-Weighted Sentiment'].mean(),  # Average VWS
            daily_return=df['Daily Return'].mean(),  # Average daily return
            rolling_avg=df[f'Rolling Avg (7 Days)'].mean(),  # Average rolling average
            volatility=df[f'Volatility (7 Days)'].mean(),  # Average volatility
            momentum=df[f'Momentum (7 Days)'].mean()  # Average momentum
        )

    # Add Industry-Based Relationships
    for ticker1 in tickers:
        for ticker2 in tickers:
            if ticker1 != ticker2:
                if feature_data[ticker1]['Industry'][0] == feature_data[ticker2]['Industry'][0]:
                    G.add_edge(ticker1, ticker2, relationship="industry")

    # Add Correlation-Based Relationships
    for ticker1 in tickers:
        for ticker2 in tickers:
            if ticker1 != ticker2:
                df1 = feature_data[ticker1]
                df2 = feature_data[ticker2]
                correlation = df1['Daily Return'].corr(df2['Daily Return'])
                if correlation and correlation > correlation_threshold:
                    G.add_edge(ticker1, ticker2, relationship="correlation", weight=correlation)

    # Add Volume-Weighted Sentiment Similarity
    for ticker1 in tickers:
        for ticker2 in tickers:
            if ticker1 != ticker2:
                vws1 = np.array(feature_data[ticker1]['Volume-Weighted Sentiment']).reshape(-1, 1)
                vws2 = np.array(feature_data[ticker2]['Volume-Weighted Sentiment']).reshape(-1, 1)
                similarity = cosine_similarity(vws1.T, vws2.T)[0, 0]
                if similarity > similarity_threshold:
                    G.add_edge(ticker1, ticker2, relationship="vws_similarity", weight=similarity)

    # Add Mutual Shareholder Relationships
    for ticker1, data1 in shareholder_data.items():
        for ticker2, data2 in shareholder_data.items():
            if ticker1 != ticker2:
                holders1 = set(data1['Holder'])
                holders2 = set(data2['Holder'])
                mutual_holders = holders1.intersection(holders2)
                if len(mutual_holders) >= mutual_shareholder_threshold:
                    G.add_edge(
                        ticker1,
                        ticker2,
                        relationship="mutual_shareholder",
                        mutual_count=len(mutual_holders)
                    )

    print(f"Graph constructed with {len(G.nodes)} nodes and {len(G.edges)} edges.")
    return G

# Save graph to file
def save_graph_to_file(graph, output_file):
    nx.write_gml(graph, output_file)
    print(f"Graph saved to {output_file}")

# Execute
graph = construct_graph_with_edges(tickers, base_path)
save_graph_to_file(graph, "stock_graph_with_edges.gml")
