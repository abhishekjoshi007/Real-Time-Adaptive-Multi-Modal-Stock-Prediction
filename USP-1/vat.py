import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json

# Dataset class
class VolatilityDataset(Dataset):
    def __init__(self, features, labels, volatility_classes):
        self.features = np.array(features, dtype=np.float32)
        self.features = np.nan_to_num(self.features, nan=0.0, posinf=0.0, neginf=0.0)
        self.labels = np.array(labels, dtype=np.float32)

        # Adjust weights based on volatility
        high_volatility = volatility_classes == "High"
        medium_volatility = volatility_classes == "Medium"
        low_volatility = volatility_classes == "Low"

        self.features[:, 0] *= np.where(high_volatility, 1.5, np.where(medium_volatility, 1.3, 1.0))  # Volume-Weighted Sentiment
        self.features[:, 2] *= np.where(low_volatility, 1.5, np.where(medium_volatility, 1.3, 1.0))  # Rolling Avg (7 Days)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y

# Improved Transformer model
class VolatilityAwareTransformer(nn.Module):
    def __init__(self, input_dim, num_heads=4, num_layers=3, hidden_dim=128, dropout=0.3):
        super(VolatilityAwareTransformer, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.transformer = nn.Transformer(
            d_model=hidden_dim, nhead=num_heads, num_encoder_layers=num_layers, dropout=dropout, batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x, volatility_embedding):
        x = self.embedding(x)
        x = x + volatility_embedding.unsqueeze(-1)  # Add volatility embedding
        x = x.unsqueeze(1)  # Add sequence dimension for Transformer
        x = self.transformer(x, x)
        x = x.mean(dim=1)  # Average over sequence length
        return self.fc(x)

# Load and preprocess your dataset
def load_data(file_path):
    data = pd.read_csv(file_path)

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

    labels = data["Close"].values  # Target: Closing price
    volatility_classes = data["Volatility Class"].values

    # Normalize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Apply PCA
    pca = PCA(n_components=min(features.shape[1], 5))  # Retain 5 principal components
    features = pca.fit_transform(features)

    return features, labels, volatility_classes, data, scaler, pca

# Main script
if __name__ == "__main__":
    # Load data
    file_path = "/Users/abhishekjoshi/Documents/GitHub/stock_forecasting_CAI/Data /WDAY/WDAY_merged_with_vix.csv"
    features, labels, volatility_classes, full_data, scaler, pca = load_data(file_path)

    # Create Dataset and DataLoader
    dataset = VolatilityDataset(features, labels, volatility_classes)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model, loss, and optimizer
    input_dim = features.shape[1]
    model = VolatilityAwareTransformer(input_dim)
    criterion = nn.SmoothL1Loss()  # Robust loss
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)  # Fine-tuned learning rate

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_features, batch_labels in dataloader:
            volatility_embedding = batch_features[:, -1].mean(dim=0, keepdim=True)

            inputs = batch_features  # All input features

            predictions = model(inputs, volatility_embedding)
            loss = criterion(predictions.view(-1), batch_labels.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    print("Training completed.")

    # Inference and evaluation metrics
    print("\nPerforming inference and evaluation...")
    model.eval()
    with torch.no_grad():
        predictions = []
        ground_truth = []
        high_volatility_errors = []
        low_volatility_errors = []

        for batch_features, batch_labels in dataloader:
            volatility_embedding = batch_features[:, -1].mean(dim=0, keepdim=True)
            preds = model(batch_features, volatility_embedding).view(-1).cpu().numpy()

            predictions.extend(preds)
            ground_truth.extend(batch_labels.cpu().numpy())

            # Convert to numpy for boolean indexing
            batch_features_np = batch_features.cpu().numpy()
            batch_labels_np = batch_labels.cpu().numpy()

            high_volatility = batch_features_np[:, 0] > 1.4
            low_volatility = batch_features_np[:, 0] < 1.1

            high_volatility_errors.extend(np.abs(preds[high_volatility] - batch_labels_np[high_volatility]))
            low_volatility_errors.extend(np.abs(preds[low_volatility] - batch_labels_np[low_volatility]))

        # Calculate evaluation metrics
        mae = mean_absolute_error(ground_truth, predictions)
        rmse = np.sqrt(mean_squared_error(ground_truth, predictions))
        sharpe_ratio = np.mean(predictions) / np.std(predictions)

        directional_accuracy = np.mean(
            np.sign(predictions) == np.sign(ground_truth)
        ) * 100

        high_volatility_error_mean = np.mean(high_volatility_errors)
        low_volatility_error_mean = np.mean(low_volatility_errors)

        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
        print(f"Directional Accuracy: {directional_accuracy:.2f}%")
        print(f"High Volatility Error Mean: {high_volatility_error_mean:.4f}")
        print(f"Low Volatility Error Mean: {low_volatility_error_mean:.4f}")

        # Save results
results = {
    "MAE": float(mae),
    "RMSE": float(rmse),
    "Sharpe Ratio": float(sharpe_ratio),
    "Directional Accuracy": float(directional_accuracy),
    "High Volatility Error Mean": float(high_volatility_error_mean),
    "Low Volatility Error Mean": float(low_volatility_error_mean),
}

with open("evaluation_results.json", "w") as f:
    json.dump(results, f)

print("Evaluation results saved to evaluation_results.json.")
