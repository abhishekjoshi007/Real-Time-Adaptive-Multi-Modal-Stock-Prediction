import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader


# Dataset class
class VolatilityDataset(Dataset):
    def __init__(self, features, labels):
        # Convert features and labels to float32 and handle NaN or infinite values
        self.features = np.array(features, dtype=np.float32)
        self.features = np.nan_to_num(self.features, nan=0.0, posinf=0.0, neginf=0.0)
        self.labels = np.array(labels, dtype=np.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


# Transformer model with volatility-aware attention
class VolatilityAwareTransformer(nn.Module):
    def __init__(self, input_dim, num_heads=4, num_layers=2, dropout=0.1):
        super(VolatilityAwareTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim + 1, 128)  # Add 1 for volatility embedding
        self.transformer = nn.Transformer(
            d_model=128, nhead=num_heads, num_encoder_layers=num_layers, dropout=dropout, batch_first=True
        )
        self.fc = nn.Linear(128, 1)  # Output single prediction (e.g., price change)

    def forward(self, x, volatility_embedding):
        # Combine inputs with volatility embedding
        x = torch.cat((x, volatility_embedding.unsqueeze(1)), dim=1)
        x = self.embedding(x)

        # Add sequence dimension (seq_len=1) for Transformer
        x = x.unsqueeze(1)  # Shape: (batch_size, seq_len, features)

        x = self.transformer(x, x)
        x = x.mean(dim=1)  # Average over sequence length
        return self.fc(x)


# Load and preprocess your dataset
def load_data(file_path):
    import pandas as pd
    data = pd.read_csv(file_path)

    # Extract features and labels
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
    return features, labels


# Main script
if __name__ == "__main__":
    # Load data
    features, labels = load_data("/Users/abhishekjoshi/Documents/GitHub/stock_forecasting_CAI/Data /PD/PD_merged_with_vix.csv")

    # Create Dataset and DataLoader
    dataset = VolatilityDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model, loss, and optimizer
    input_dim = features.shape[1] - 1  # Exclude volatility column from input dim
    model = VolatilityAwareTransformer(input_dim)
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_features, batch_labels in dataloader:
            # Separate features and volatility embedding
            volatility_embedding = batch_features[:, -1]  # Extract volatility
            inputs = batch_features[:, :-1]  # Exclude volatility from main input

            # Forward pass
            predictions = model(inputs, volatility_embedding)
            loss = criterion(predictions.view(-1), batch_labels.view(-1))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader):.4f}")
