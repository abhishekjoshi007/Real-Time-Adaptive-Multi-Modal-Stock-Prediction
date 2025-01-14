import networkx as nx
import numpy as np
import torch
from torch_geometric.nn import Node2Vec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the GML File
gml_file = "/Users/abhishekjoshi/Documents/GitHub/stock_forecasting_CAI/USP-2/stock_graph_with_edges.gml"
graph = nx.read_gml(gml_file)

# Convert the NetworkX graph to PyTorch Geometric format
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

    return x, edge_index, y

data_x, edge_index, data_y = nx_to_pyg_data(graph)

# Preprocess features and labels
scaler = StandardScaler()
data_x = torch.tensor(scaler.fit_transform(data_x.numpy()), dtype=torch.float)
data_y = (data_y - data_y.mean()) / data_y.std()  # Standardize labels

# Corrected split_data function
def split_data(data, train_ratio=0.7, val_ratio=0.15):
    indices = np.arange(data.shape[0])
    train_idx, temp_idx = train_test_split(indices, test_size=(1 - train_ratio), random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=val_ratio / (1 - train_ratio), random_state=42)

    return torch.tensor(train_idx, dtype=torch.long), torch.tensor(val_idx, dtype=torch.long), torch.tensor(test_idx, dtype=torch.long)

data_train_idx, data_val_idx, data_test_idx = split_data(data_x)

# Define and initialize the Node2Vec model
node2vec = Node2Vec(edge_index=edge_index, embedding_dim=16, walk_length=10, context_size=5, walks_per_node=10, num_negative_samples=1, p=1, q=1, sparse=True)

# Train Node2Vec model
def train_node2vec(model, data_x, epochs=100, lr=0.01):
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        batch = torch.arange(data_x.shape[0], device=data_x.device)  # Provide a batch of nodes
        pos_rw = model.pos_sample(batch)  # Generate positive random walks
        neg_rw = model.neg_sample(batch)  # Generate negative random walks
        loss = model.loss(pos_rw, neg_rw)  # Use model's internal loss function
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    return model

# Train the Node2Vec model
trained_node2vec = train_node2vec(node2vec, data_x, epochs=100, lr=0.01)

# Get node embeddings
embeddings = trained_node2vec.forward()

# Prepare embeddings for downstream regression
def prepare_data(embeddings, labels, train_idx, val_idx, test_idx):
    X_train, X_val, X_test = embeddings[train_idx].detach().numpy(), embeddings[val_idx].detach().numpy(), embeddings[test_idx].detach().numpy()
    y_train, y_val, y_test = labels[train_idx].numpy(), labels[val_idx].numpy(), labels[test_idx].numpy()
    return X_train, X_val, X_test, y_train, y_val, y_test

X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(embeddings, data_y, data_train_idx, data_val_idx, data_test_idx)

# Train a regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Evaluate the model
y_pred = regressor.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

directional_accuracy = np.mean((np.sign(y_pred[1:] - y_pred[:-1]) == np.sign(y_test[1:] - y_test[:-1])))

returns = y_pred - y_test
sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) != 0 else 0

ic = np.corrcoef(y_pred, y_test)[0, 1]

# Calculate hit rate
hit_rate = np.mean(np.sign(y_pred) == np.sign(y_test))

# Add hit rate to metrics
metrics = {
    "MAE": mae,
    "RMSE": rmse,
    "MAPE (%)": mape,
    "Directional Accuracy (%)": directional_accuracy * 100,
    "Sharpe Ratio": sharpe_ratio,
    "Information Coefficient (IC)": ic,
    "Hit Rate (%)": hit_rate * 100,
}

# Display evaluation metrics
print("Evaluation Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")
