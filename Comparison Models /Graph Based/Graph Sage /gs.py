import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the GML File
gml_file = "/Users/abhishekjoshi/Documents/GitHub/stock_forecasting_CAI/USP-2/stock_graph_with_edges.gml"
graph = nx.read_gml(gml_file)

# Convert the NetworkX graph to PyTorch Geometric Data
def nx_to_pyg_data(graph):
    node_features = []
    node_labels = []
    node_mapping = {}

    for i, (node, data) in enumerate(graph.nodes(data=True)):
        # Extract node features
        node_features.append([
            float(data.get('volume_weighted_sentiment', 0)),
            float(data.get('daily_return', 0)),
            float(data.get('rolling_avg', 0)),
            float(data.get('volatility', 0)),
            float(data.get('momentum', 0)),
        ])
        # Use daily return as the target label
        node_labels.append(float(data.get('daily_return', 0)))
        node_mapping[node] = i

    # Convert edges
    edge_index = []
    for src, dst in graph.edges():
        edge_index.append([node_mapping[src], node_mapping[dst]])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor(node_labels, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, y=y)

data = nx_to_pyg_data(graph)

# Preprocess features and labels
scaler = StandardScaler()
data.x = torch.tensor(scaler.fit_transform(data.x.numpy()), dtype=torch.float)
data.y = (data.y - data.y.mean()) / data.y.std()  # Standardize labels


# Corrected split_data function
def split_data(data, train_ratio=0.7, val_ratio=0.15):
    indices = np.arange(data.num_nodes)
    # Split into train and temporary set
    train_idx, temp_idx = train_test_split(indices, test_size=(1 - train_ratio), random_state=42)
    # Adjust the validation split size to be proportional to the temporary set
    val_split = val_ratio / (1 - train_ratio)
    val_idx, test_idx = train_test_split(temp_idx, test_size=(1 - val_split), random_state=42)

    return torch.tensor(train_idx, dtype=torch.long), torch.tensor(val_idx, dtype=torch.long), torch.tensor(test_idx, dtype=torch.long)

# Usage in main script
data.train_idx, data.val_idx, data.test_idx = split_data(data)


# Define the GraphSAGE model
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

# Train the GraphSAGE model
def train_model(model, data, epochs=100, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = torch.nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(data.x, data.edge_index).squeeze()
        loss = criterion(output[data.train_idx], data.y[data.train_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(output[data.val_idx], data.y[data.val_idx]).item()

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")

    return model

# Evaluate the model with Hit Rate included
def evaluate_model(model, data):
    model.eval()
    predictions = model(data.x, data.edge_index).squeeze().detach().numpy()
    true_values = data.y.detach().numpy()

    # Calculate metrics
    mae = mean_absolute_error(true_values[data.test_idx], predictions[data.test_idx])
    rmse = np.sqrt(mean_squared_error(true_values[data.test_idx], predictions[data.test_idx]))
    mape = np.mean(np.abs((true_values[data.test_idx] - predictions[data.test_idx]) / true_values[data.test_idx])) * 100

    directional_accuracy = np.mean(
        (np.sign(predictions[data.test_idx][1:] - predictions[data.test_idx][:-1]) ==
         np.sign(true_values[data.test_idx][1:] - true_values[data.test_idx][:-1]))
    )

    returns = predictions[data.test_idx] - true_values[data.test_idx]
    sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) != 0 else 0

    ic = np.corrcoef(predictions[data.test_idx], true_values[data.test_idx])[0, 1]

    # Calculate Hit Rate
    threshold = 0.01  # Define a threshold for considering a prediction as "hit"
    hits = np.sum(np.abs(predictions[data.test_idx] - true_values[data.test_idx]) <= threshold)
    hit_rate = (hits / len(data.test_idx)) * 100

    # Return metrics
    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape,
        "Directional Accuracy (%)": directional_accuracy * 100,
        "Sharpe Ratio": sharpe_ratio,
        "Information Coefficient (IC)": ic,
        "Hit Rate (%)": hit_rate,
    }

    return metrics

# Main execution
in_channels = data.x.shape[1]
hidden_channels = 16
out_channels = 1

model = GraphSAGE(in_channels, hidden_channels, out_channels)
trained_model = train_model(model, data, epochs=100, lr=0.01)
metrics = evaluate_model(trained_model, data)

print("Evaluation Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")
