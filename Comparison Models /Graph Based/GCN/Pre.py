import os
import pandas as pd
import networkx as nx
import json
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Paths
data_dir = '/Users/abhishekjoshi/Documents/GitHub/stock_forecasting_CAI/Merged Data'
usp4_file = '/Users/abhishekjoshi/Documents/GitHub/stock_forecasting_CAI/USP 4/daily_recommendations.csv'

# Initialize an empty graph
G = nx.Graph()

# Load USP4 recommendations
if os.path.exists(usp4_file):
    usp4_data = pd.read_csv(usp4_file)
else:
    raise FileNotFoundError(f"USP4 file not found at {usp4_file}")

# Helper function for volatility calculation
def calculate_volatility(prices):
    return np.std(np.diff(np.log(prices)))

# Function to clean data
def clean_data(series):
    return series.replace([np.inf, -np.inf], np.nan).dropna()

# Function to calculate daily returns
def calculate_daily_returns(prices):
    return np.diff(prices) / prices[:-1]

# Function to calculate cosine similarity
def cosine_similarity_calc(vec1, vec2):
    vec1 = vec1.reshape(1, -1)
    vec2 = vec2.reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]

# Parameters
correlation_threshold = 0.8
similarity_threshold = 0.9
mutual_shareholder_threshold = 5

# Iterate through all ticker folders in the data directory
historical_data_dict = {}
holder_data_dict = {}

for ticker_folder in os.listdir(data_dir):
    ticker_path = os.path.join(data_dir, ticker_folder)

    # Skip if not a directory
    if not os.path.isdir(ticker_path):
        continue

    # Load historical data
    csv_file = os.path.join(ticker_path, f"{ticker_folder}.csv")
    if os.path.exists(csv_file):
        historic_data = pd.read_csv(csv_file)
        historical_data_dict[ticker_folder] = historic_data
    else:
        print(f"Historical data missing for {ticker_folder}. Skipping.")
        continue

    # Extract industry information
    industry = historic_data.iloc[0]["Industry"]  # Assuming the industry is consistent for a ticker

    # Load holders data
    holder_file = os.path.join(ticker_path, f"{ticker_folder}_holder.json")
    if os.path.exists(holder_file):
        with open(holder_file, 'r') as f:
            holder_data = json.load(f)
            holder_data_dict[ticker_folder] = holder_data
    else:
        print(f"Holders data missing for {ticker_folder}. Skipping.")
        continue

    # Extract node features from historical data
    latest_data = historic_data.iloc[-1].to_dict()
    node_features = {
        "close_price": latest_data.get("Close"),
        "volume": latest_data.get("Volume"),
        "high": latest_data.get("High"),
        "low": latest_data.get("Low"),
        "adj_close": latest_data.get("Adj Close"),
        "industry": industry,
    }

    # Add holder-specific data
    total_holders = len(holder_data) if isinstance(holder_data, list) else 0
    node_features["total_holders"] = total_holders

    # Add node to the graph
    G.add_node(ticker_folder, **node_features)

# Add edges based on the criteria
tickers = list(historical_data_dict.keys())

for i, ticker1 in enumerate(tickers):
    for j, ticker2 in enumerate(tickers):
        if i >= j:
            continue

        # Industry-Based Relationships
        industry1 = G.nodes[ticker1].get("industry")
        industry2 = G.nodes[ticker2].get("industry")
        if industry1 and industry2 and industry1 == industry2:
            G.add_edge(ticker1, ticker2, relationship="industry")

        # Correlation-Based Relationships
        prices1 = clean_data(historical_data_dict[ticker1]["Close"])
        prices2 = clean_data(historical_data_dict[ticker2]["Close"])
        if len(prices1) == len(prices2):
            daily_returns1 = calculate_daily_returns(prices1.values)
            daily_returns2 = calculate_daily_returns(prices2.values)
            corr, _ = pearsonr(daily_returns1, daily_returns2)
            if corr > correlation_threshold:
                G.add_edge(ticker1, ticker2, relationship="correlation", weight=corr)

        # Volume-Weighted Sentiment Similarity
        vws1 = clean_data(historical_data_dict[ticker1]["Volume-Weighted Sentiment"])
        vws2 = clean_data(historical_data_dict[ticker2]["Volume-Weighted Sentiment"])
        if len(vws1) == len(vws2):
            similarity = cosine_similarity_calc(vws1.values, vws2.values)
            if similarity > similarity_threshold:
                G.add_edge(ticker1, ticker2, relationship="vws_similarity", weight=similarity)

        # Mutual Shareholder Relationships
        holders1 = set(holder.get("name") for holder in holder_data_dict.get(ticker1, []))
        holders2 = set(holder.get("name") for holder in holder_data_dict.get(ticker2, []))
        common_holders = holders1.intersection(holders2)
        if len(common_holders) >= mutual_shareholder_threshold:
            G.add_edge(ticker1, ticker2, relationship="mutual_shareholder", mutual_count=len(common_holders))

# Add edges based on USP4 recommendations
for _, row in usp4_data.iterrows():
    ticker1 = row.get('Ticker1')  # Replace with actual column name
    ticker2 = row.get('Ticker2')  # Replace with actual column name
    recommendation_score = row.get('Score', 0)  # Replace with actual column name

    if ticker1 in G.nodes and ticker2 in G.nodes:
        G.add_edge(ticker1, ticker2, weight=recommendation_score)

# Print graph summary
print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

# Save the graph to a file
output_graph_file = "output_graph_with_criteria.graphml"
nx.write_graphml(G, output_graph_file)
print(f"Graph saved to {output_graph_file}")

# Visualization (optional)
plt.figure(figsize=(12, 8))
nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
plt.title("Graph Representation of Stock Data")
plt.show()
