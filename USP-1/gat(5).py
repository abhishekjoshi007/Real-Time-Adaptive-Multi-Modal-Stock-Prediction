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

# Graph Attention Network model
class GraphAttentionNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_heads=4, dropout=0.3):
        super(GraphAttentionNetwork, self).__init__()

        # Adjust input_dim to be divisible by num_heads
        adjusted_input_dim = (input_dim // num_heads) * num_heads
        if adjusted_input_dim != input_dim:
            print(f"Adjusting input_dim from {input_dim} to {adjusted_input_dim} to be divisible by num_heads={num_heads}")
        
        self.input_dim = adjusted_input_dim

        # Multihead Attention layer
        self.gat_layer = nn.MultiheadAttention(embed_dim=self.input_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # Ensure x is reshaped correctly for attention layer
        batch_size, num_features = x.shape
        if num_features != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, but got {num_features}")
        
        x = x.unsqueeze(1)  # Add a sequence length dimension (sequence length = 1)
        attention_output, _ = self.gat_layer(x, x, x)
        attention_output = attention_output.mean(dim=1)  # Average over nodes
        return self.fc(attention_output)


# Load and preprocess your dataset
def load_data(file_path, target_dim=4):
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
    max_components = min(features.shape[0], features.shape[1], target_dim)  # Ensure target_dim is used
    pca = PCA(n_components=max_components)  # Use fixed target_dim
    features = pca.fit_transform(features)

    # Ensure the final dimension is divisible by num_heads
    if features.shape[1] % 4 != 0:
        padded_dim = (features.shape[1] // 4 + 1) * 4
        features = np.pad(features, ((0, 0), (0, padded_dim - features.shape[1])), mode='constant')

    return features, labels, volatility_classes, data, scaler, pca

# Main script
if __name__ == "__main__":
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
        'VSAT', 'BASE', 'MAXN', 'PGY', 'TDY', 'PAR', 'MRVL', 'PLUS', 'GCT', 'BKSY', 'FOUR'
    ]

    for ticker in tickers:
        try:
            print(f"Processing ticker: {ticker}")
            file_path = f"/Users/gurojaschadha/Downloads/data_vw_sentiment/{ticker}/{ticker}_merged_with_vix.csv"
            features, labels, volatility_classes, full_data, scaler, pca = load_data(file_path, target_dim=4)

            # Create Dataset and DataLoader
            dataset = VolatilityDataset(features, labels, volatility_classes)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

            # Initialize model, loss, and optimizer
            input_dim = features.shape[1]
            model = GraphAttentionNetwork(input_dim)
            criterion = nn.SmoothL1Loss()  # Robust loss
            optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)  # Fine-tuned learning rate

            # Training loop
            num_epochs = 100
            for epoch in range(num_epochs):
                model.train()
                epoch_loss = 0.0
                for batch_features, batch_labels in dataloader:
                    inputs = batch_features  # All input features

                    predictions = model(inputs)
                    loss = criterion(predictions.view(-1), batch_labels.view(-1))

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

            print(f"Training completed for {ticker}.")

            # Inference and evaluation metrics
            print(f"Performing inference and evaluation for {ticker}...")
            model.eval()
            with torch.no_grad():
                predictions = []
                ground_truth = []

                for batch_features, batch_labels in dataloader:
                    preds = model(batch_features).view(-1).cpu().numpy()

                    predictions.extend(preds)
                    ground_truth.extend(batch_labels.cpu().numpy())

                # Calculate evaluation metrics
                mae = mean_absolute_error(ground_truth, predictions)
                rmse = np.sqrt(mean_squared_error(ground_truth, predictions))
                sharpe_ratio = np.mean(predictions) / np.std(predictions)
                directional_accuracy = np.mean(np.sign(predictions) == np.sign(ground_truth)) * 100

                # Save results
                results = {
                    "Ticker": ticker,
                    "MAE": float(mae),
                    "RMSE": float(rmse),
                    "Sharpe Ratio": float(sharpe_ratio),
                    "Directional Accuracy": float(directional_accuracy),
                }

                output_file = f"{ticker}/evaluation_results_{ticker}.json"
                with open(output_file, "w") as f:
                    json.dump(results, f)

                print(f"Evaluation results saved for {ticker} in {output_file}.")
        except Exception as e:
            print(f"Error processing ticker {ticker}: {e}")