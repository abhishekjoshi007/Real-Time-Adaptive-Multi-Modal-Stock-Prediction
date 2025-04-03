import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import json
import os


# 1. DATASET DEFINITION
class VolatilityGraphDataset(Dataset):
    """
    Dataset that loads features (inc. volatility info), labels, volatility classes,
    AND a per-sample sub-adjacency or graph embedding index.

    For simplicity, this example reads an adjacency matrix from a single graph file
    (one graph for all data). If each sample is a single node, you might store the
    row from adjacency as part of x, or store node-IDs to do a mini-batch gather, etc.
    """
    def __init__(self, features, labels, volatility_classes, adjacency):
        """
        :param features:       N x D array of features
        :param labels:         N array of labels
        :param volatility_classes: N array of "High"/"Medium"/"Low"
        :param adjacency:      Full adjacency matrix or some graph structure
        """
        # Features
        self.features = np.array(features, dtype=np.float32)
        self.features = np.nan_to_num(self.features, nan=0.0, posinf=0.0, neginf=0.0)
        # Labels
        self.labels = np.array(labels, dtype=np.float32)
        # Volatility classes
        self.volatility_classes = volatility_classes
        # Graph adjacency (or partial)
        self.adjacency = adjacency  # Possibly a 2D np.array if you want

        # "Volume-Weighted Sentiment" => col 0
        # "Rolling Avg (7 Days)"      => col 2
        high_volatility = (self.volatility_classes == "High")
        medium_volatility = (self.volatility_classes == "Medium")
        low_volatility = (self.volatility_classes == "Low")

        # Reweight col 0 based on high/medium/low
        self.features[:, 0] *= np.where(high_volatility, 1.5, 
                                 np.where(medium_volatility, 1.3, 1.0))
        # Reweight col 2 based on high/medium/low
        self.features[:, 2] *= np.where(low_volatility, 1.5, 
                                 np.where(medium_volatility, 1.3, 1.0))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        """
        Returns the features, label, and adjacency or adjacency sub-vector for node idx.
        Here, we assume the entire adjacency matrix is used in the model (like a full GNN),
        so we won't pass adjacency for a single node only. Alternatively, you could pass row 'idx'.
        """
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y, idx  # We return idx to let the model gather adjacency rows if needed
# 2. MODEL DEFINITION
class VolatilityAwareTransformerWithGraph(nn.Module):
    """
    Example model that:
      1) Takes input features x
      2) Gathers adjacency for that mini-batch from a global adjacency matrix
      3) Produces a 'graph embedding' (toy example) 
      4) Feeds x + adjacency-based embedding + volatility embedding into Transformer
    """
    def __init__(self, input_dim, num_heads=4, num_layers=3, hidden_dim=128, dropout=0.3, adjacency=None):
        super().__init__()
        self.adjacency = adjacency  # Full adjacency matrix (N x N), if relevant
        self.n_nodes = adjacency.shape[0] if adjacency is not None else 0

        # Basic feed-forward to embed adjacency
        # In real use, you'd do something more sophisticated (GraphConv, GAT, etc.)
        self.adj_embed_dim = 16  # or any dimension you like
        self.adj_embedding = nn.Linear(self.n_nodes, self.adj_embed_dim)

        # We'll embed the main features
        self.feature_embedding_dim = hidden_dim
        self.feature_embedding = nn.Sequential(
            nn.Linear(input_dim + self.adj_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        # The actual Transformer
        self.transformer = nn.Transformer(
            d_model=hidden_dim, nhead=num_heads,
            num_encoder_layers=num_layers, dropout=dropout, batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x, volatility_embedding, batch_indices):
        """
        x: (batch_size, input_dim)
        volatility_embedding: (batch_size, 1) or aggregated dimension
        batch_indices: (batch_size,) => node indices for adjacency usage
        """
        batch_size = x.size(0)

        # 1) Gather adjacency rows for these nodes
        # adjacency shape => (N, N)
        # We want adjacency for each index in batch_indices
        # shape => (batch_size, N)
        if self.adjacency is not None:
            device = x.device
            adjacency_subset = []
            for idx in batch_indices:
                row = self.adjacency[idx.item()]  # row is shape (N,)
                adjacency_subset.append(row)
            adjacency_subset = np.stack(adjacency_subset, axis=0)  # shape => (batch_size, N)
            adjacency_subset = torch.tensor(adjacency_subset, dtype=torch.float32, device=device)
        else:
            # fallback if no adjacency
            adjacency_subset = torch.zeros(batch_size, 1, device=x.device)

        # 2) Embed adjacency => shape (batch_size, adj_embed_dim)
        adj_embed = self.adj_embedding(adjacency_subset)

        # 3) Combine adjacency embedding + main features => shape (batch_size, input_dim + adj_embed_dim)
        combined_input = torch.cat([x, adj_embed], dim=1)

        # 4) Pass combined input through feature embedding => shape (batch_size, hidden_dim)
        out = self.feature_embedding(combined_input)

        # 5) Add volatility embedding
        # shape volatility_embedding => (batch_size, 1)
    
        vol_expanded = volatility_embedding.repeat(1, out.size(1))  # shape => (batch_size, hidden_dim)
        out = out + vol_expanded

        # 6) Reshape for Transformer => (batch_size, seq_len=1, hidden_dim)
        out = out.unsqueeze(1)

        # 7) Pass through Transformer
        # We'll pass the same data as src and tgt if we treat it as an 'encoder' only scenario
        out = self.transformer(out, out)  # shape => (batch_size, seq_len, hidden_dim)

        # 8) Pool or reduce => shape => (batch_size, hidden_dim)
        # if seq_len=1, mean over dim=1 is the same as out[:,0,:]
        out = out.mean(dim=1)

        # 9) Final FC => shape => (batch_size, 1)
        out = self.fc(out)
        return out
# 3. LOAD GRAPH + DA
def load_data_and_graph(historic_file, graph_file):
    """
    Loads your historic data, returns features, labels, volatility classes,
    plus adjacency from graph_file
    """
    # Load CSV
    data = pd.read_csv(historic_file)

    features = data[
        [
            "Volume-Weighted Sentiment",
            "Normalized VWS (7 Days)",
            "Rolling Avg (7 Days)",
            "EWMA Volatility",
            "Interest_Rate",
            "Inflation",
            "GDP",
        ]
    ].values

    labels = data["Close"].values  # Target
    volatility_classes = data["Volatility Class"].values

    # Load adjacency from graph file
    adjacency_df = pd.read_csv(graph_file, header=None)
    adjacency = adjacency_df.values  # shape => (N, N)

    # scale features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # pca
    pca = PCA(n_components=min(features.shape[1], 5))
    features = pca.fit_transform(features)

    return features, labels, volatility_classes, adjacency, data, scaler, pca
# 4. TRAINING / EVALUATI
if __name__ == "__main__":

    graph_file = "/Users/abhishekjoshi/Documents/GitHub/stock_forecasting_CAI/USP-2/stock_graph_with_edges.gml"

    # Tickers
    tickers = ["AAPL","GOOG"]  # Example
    cumulative_metrics = {
        "MAE": [],
        "RMSE": [],
        "Sharpe Ratio": [],
        "Directional Accuracy": [],
        "IC": []
    }

    for ticker in tickers:
        try:
            print(f"\nProcessing Ticker: {ticker}")
            historic_file = f"/my/path/{ticker}_merged_with_vix.csv"
            features, labels, vol_classes, adjacency, full_data, scaler, pca = load_data_and_graph(historic_file, graph_file)

            # Create dataset / dataloader
            dataset = VolatilityGraphDataset(features, labels, vol_classes, adjacency)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

            input_dim = features.shape[1]  # after PCA
            adjacency_tensor = torch.tensor(adjacency, dtype=torch.float32)  # for the model

            # Init Model
            model = VolatilityAwareTransformerWithGraph(
                input_dim=input_dim,
                num_heads=4,
                num_layers=3,
                hidden_dim=128,
                dropout=0.3,
                adjacency=adjacency_tensor
            )

            criterion = nn.SmoothL1Loss()
            optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)

            # Train
            num_epochs = 200
            for epoch in range(num_epochs):
                model.train()
                epoch_loss = 0.0
                for batch_idx, (batch_features, batch_labels, batch_indices) in enumerate(dataloader):

                    # For volatility embedding, we do a simple mean or something
                    volatility_embedding = batch_features[:, -1].mean(dim=0, keepdim=True).unsqueeze(0)
                    # Actually, we want shape => (batch_size, 1)
                    # So let's do the mean per sample's last feature or not:
                    # We'll do it the same as the simpler code:
                    vol_embed = batch_features[:, -1].unsqueeze(1)  # shape => (batch_size,1)

                    preds = model(batch_features, vol_embed, batch_indices)
                    loss = criterion(preds.view(-1), batch_labels.view(-1))

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                avg_loss = epoch_loss / len(dataloader)
                print(f"Epoch [{epoch+1}/{num_epochs}], {ticker}, Avg Loss: {avg_loss:.4f}")

            # Inference
            model.eval()
            predictions = []
            ground_truth = []
            with torch.no_grad():
                for (batch_features, batch_labels, batch_indices) in dataloader:
                    vol_embed = batch_features[:, -1].unsqueeze(1)
                    preds = model(batch_features, vol_embed, batch_indices).view(-1).cpu().numpy()
                    predictions.extend(preds)
                    ground_truth.extend(batch_labels.numpy())

            # Evaluate
            mae = mean_absolute_error(ground_truth, predictions)
            rmse = np.sqrt(mean_squared_error(ground_truth, predictions))
            sharpe_ratio = np.mean(predictions) / np.std(predictions) if np.std(predictions)!=0 else 0.0
            direction_acc = np.mean(np.sign(predictions) == np.sign(ground_truth))*100
            ic, _ = pearsonr(predictions, ground_truth)

            # Save
            cumulative_metrics["MAE"].append(mae)
            cumulative_metrics["RMSE"].append(rmse)
            cumulative_metrics["Sharpe Ratio"].append(sharpe_ratio)
            cumulative_metrics["Directional Accuracy"].append(direction_acc)
            cumulative_metrics["IC"].append(ic)

            print(f"\nResults for {ticker}:")
            print(f"  MAE = {mae:.4f}")
            print(f"  RMSE = {rmse:.4f}")
            print(f"  Sharpe Ratio = {sharpe_ratio:.4f}")
            print(f"  Directional Accuracy = {direction_acc:.2f}%")
            print(f"  IC = {ic:.4f}")

            out_dict = {
                "Ticker": ticker,
                "MAE": mae,
                "RMSE": rmse,
                "Sharpe Ratio": sharpe_ratio,
                "Directional Accuracy": direction_acc,
                "IC": ic
            }

            with open(f"evaluation_results_{ticker}.json","w") as f:
                json.dump(out_dict, f)

        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    # Summaries
    mean_mae = float(np.mean(cumulative_metrics["MAE"]))
    mean_rmse = float(np.mean(cumulative_metrics["RMSE"]))
    mean_sharpe = float(np.mean(cumulative_metrics["Sharpe Ratio"]))
    mean_dir_acc = float(np.mean(cumulative_metrics["Directional Accuracy"]))
    mean_ic = float(np.mean(cumulative_metrics["IC"]))

    results_summary = {
        "Average MAE": mean_mae,
        "Average RMSE": mean_rmse,
        "Average Sharpe Ratio": mean_sharpe,
        "Average Directional Accuracy": mean_dir_acc,
        "Average IC": mean_ic
    }

    with open("cumulative_graph_transformer_results.json","w") as f:
        json.dump(results_summary, f)

    print("\nCumulative results with Graph-based Volatility Transformer:")
    for k, v in results_summary.items():
        print(f"{k}: {v:.4f}")
