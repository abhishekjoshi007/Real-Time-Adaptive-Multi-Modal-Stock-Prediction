import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import networkx as nx
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Path to the GML file
gml_file = "stock_graph_with_edges.gml"

# Step 1: Load the Graph from GML File
def load_graph_from_gml(gml_file):
    graph = nx.read_gml(gml_file)
    edge_index = []
    features = []
    labels = []

    # Create a mapping from ticker to integer node ID
    ticker_to_id = {node: idx for idx, node in enumerate(graph.nodes)}

    # Extract node features
    for node, data in graph.nodes(data=True):
        features.append([
            data.get('volume_weighted_sentiment', 0),
            data.get('daily_return', 0),
            data.get('rolling_avg', 0),
            data.get('volatility', 0),
            data.get('momentum', 0)
        ])
        labels.append(data.get('target', 1))  # Adjust based on your dataset

    # Extract edges and map tickers to integer IDs
    for edge in graph.edges(data=True):
        edge_index.append([ticker_to_id[edge[0]], ticker_to_id[edge[1]]])

    return np.array(features), edge_index, np.array(labels)

# Step 2: Preprocess Features
def preprocess_features(features):
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features)
    return torch.tensor(normalized_features, dtype=torch.float)

# Step 3: Create PyG Data Object
def create_pyg_data(features, edge_index, labels):
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    labels = torch.tensor(labels, dtype=torch.float).view(-1, 1)
    data = Data(x=features, edge_index=edge_index, y=labels)
    return data

# Step 4: Define GAT Model
class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=1):
        super(GAT, self).__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads)
        self.gat2 = GATConv(hidden_dim * heads, output_dim, heads=1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        return x

# Step 5: Train/Test Split
def split_data(data):
    num_nodes = data.num_nodes
    indices = list(range(num_nodes))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=42)

    data.train_mask = torch.tensor(train_idx, dtype=torch.long)
    data.val_mask = torch.tensor(val_idx, dtype=torch.long)
    data.test_mask = torch.tensor(test_idx, dtype=torch.long)
    return data

# Step 6: Train the GAT Model
def train_model(data, model, epochs=200, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    best_val_loss = float('inf')
    early_stopping_counter = 0
    patience = 20

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.mse_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        val_loss = F.mse_loss(out[data.val_mask], data.y[data.val_mask])
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Val Loss: {val_loss.item()}")

        # Early stopping
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            print("Early stopping...")
            break

# Step 7: Test the Model
def test_model(data, model):
    model.eval()
    out = model(data)
    test_loss = F.mse_loss(out[data.test_mask], data.y[data.test_mask])
    print(f"Test Loss: {test_loss.item()}")
    return out

# Evaluate Metrics (Enhanced for IC Calculation)
def evaluate_metrics(data, predictions):
    y_true = data.y[data.test_mask].detach().cpu().numpy()
    y_pred = predictions[data.test_mask].detach().cpu().numpy()

    # Calculate MAE and RMSE
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)

    # Calculate IC (Handle zero variance)
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        ic = 0  # Set IC to 0 if variance is zero
    else:
        ic = np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]

    # Calculate Directional Accuracy
    correct_directions = (np.sign(y_true) == np.sign(y_pred)).sum()
    directional_accuracy = correct_directions / len(y_true)

    return mae, rmse, ic, directional_accuracy

# Save Results to a File
def save_results(mae, rmse, ic, directional_accuracy, output_file="results.txt"):
    with open(output_file, "w") as file:
        file.write("Model Evaluation Results\n")
        file.write("-------------------------\n")
        file.write(f"Mean Absolute Error (MAE): {mae:.6f}\n")
        file.write(f"Root Mean Squared Error (RMSE): {rmse:.6f}\n")
        file.write(f"Information Coefficient (IC): {ic:.6f}\n")
        file.write(f"Directional Accuracy: {directional_accuracy:.6f}\n")
    print(f"Results saved to {output_file}")

# Main Execution
def main():
    # Load graph from GML file
    features, edge_index, labels = load_graph_from_gml(gml_file)

    # Preprocess features and create PyG data object
    features = preprocess_features(features)
    pyg_data = create_pyg_data(features, edge_index, labels)

    # Split data for training/testing
    pyg_data = split_data(pyg_data)

    # Initialize and train the GAT model
    model = GAT(input_dim=features.shape[1], hidden_dim=16, output_dim=1, heads=2)
    train_model(pyg_data, model)

    # Test the model
    predictions = test_model(pyg_data, model)

    # Evaluate metrics
    mae, rmse, ic, directional_accuracy = evaluate_metrics(pyg_data, predictions)
    print(f"MAE: {mae}, RMSE: {rmse}, IC: {ic}, Directional Accuracy: {directional_accuracy}")

    # Save results to file
    save_results(mae, rmse, ic, directional_accuracy)

if __name__ == "__main__":
    main()
