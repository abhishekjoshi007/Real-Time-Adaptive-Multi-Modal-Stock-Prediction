#event aware transformer model 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

class EventAwareAttention(nn.Module):
    def __init__(self, hidden_dim, dk):
        super(EventAwareAttention, self).__init__()
        self.query_layer = nn.Linear(hidden_dim, hidden_dim)
        self.key_layer = nn.Linear(hidden_dim, hidden_dim)
        self.value_layer = nn.Linear(hidden_dim, hidden_dim)
        self.event_weight = nn.Parameter(torch.zeros(1))  # Learnable λ parameter
        self.shape_ratio = nn.Parameter(torch.ones(1))  # Learnable shape ratio parameter
        self.dk = dk  # Dimensionality of the key vector

        # Metrics to track
        self.total_bidirectional_mean = 0
        self.total_bidirectional_std = 0
        self.total_shape_ratio = 0
        self.count = 0

    def forward(self, X, event_flag, lengths):
        # Compute Q, K, V
        Q = self.query_layer(X)  # (batch_size, seq_len, hidden_dim)
        K = self.key_layer(X)  # (batch_size, seq_len, hidden_dim)
        V = self.value_layer(X)  # (batch_size, seq_len, hidden_dim)

        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.dk ** 0.5)  # (batch_size, seq_len, seq_len)

        # Add bidirectional adjustment
        bidirectional_scores = attention_scores * self.shape_ratio  # Shape ratio scales attention bidirectionally

        # Log metrics
        self.total_bidirectional_mean += bidirectional_scores.mean().item()
        self.total_bidirectional_std += bidirectional_scores.std().item()
        self.total_shape_ratio += self.shape_ratio.item()
        self.count += 1

        # Ensure event_flag shape matches (batch_size, seq_len, 1)
        event_flag = event_flag.squeeze(-1) if event_flag.dim() == 4 else event_flag
        event_flag = event_flag.unsqueeze(-1) if event_flag.dim() == 2 else event_flag  # (batch_size, seq_len, 1)

        # Expand event_flag to match attention_scores
        event_flag_expanded = event_flag.expand(-1, -1, bidirectional_scores.size(-1))  # (batch_size, seq_len, seq_len)

        # Add event-aware adjustment
        attention_scores = bidirectional_scores + self.event_weight * event_flag_expanded
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, seq_len, seq_len)

        # Apply attention weights
        context = torch.matmul(attention_weights, V)  # (batch_size, seq_len, hidden_dim)

        return context, attention_weights

    def compute_overall_metrics(self):
        # Compute averages
        avg_bidirectional_mean = self.total_bidirectional_mean / self.count
        avg_bidirectional_std = self.total_bidirectional_std / self.count
        avg_shape_ratio = self.total_shape_ratio / self.count

        return avg_bidirectional_mean, avg_bidirectional_std, avg_shape_ratio


# Define the Full Stock Prediction Model
class StockPredictionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dk):
        super(StockPredictionModel, self).__init__()
        
        # Input Embedding Layer
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Event-Aware Attention Layer with Bidirectional Support
        self.attention_layer = EventAwareAttention(hidden_dim, dk)
        
        # Dropout Layer for Regularization
        self.dropout = nn.Dropout(p=0.2)
        
        # Final Prediction Layer
        self.prediction_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, X, event_flag, lengths):
        # Input Embedding
        embedded = self.input_embedding(X)  # (batch_size, seq_len, hidden_dim)
        
        # Attention Layer
        context, attention_weights = self.attention_layer(embedded, event_flag, lengths)
        
        # Apply Dropout
        context = self.dropout(context)
        
        # Prediction Layer
        predictions = self.prediction_layer(context)  # (batch_size, seq_len, output_dim)
        
        return predictions, attention_weights


# Data Preparation
def load_data(data_dir):
    features_list = []
    event_flags_list = []
    targets_list = []
    lengths_list = []

    scaler = MinMaxScaler()

    for ticker_folder in os.listdir(data_dir):
        ticker_path = os.path.join(data_dir, ticker_folder)
        input_file = os.path.join(ticker_path, f"{ticker_folder}_usp3_prepared_data.csv")
        
        if os.path.exists(input_file):
            # Load processed data
            data = pd.read_csv(input_file)

            # Standardize column names
            data.columns = data.columns.str.replace(' ', '_').str.lower()

            # Create 'Next Day Close' and 'Next Day Volatility'
            data['next_day_close'] = data['close'].shift(-1)
            data['next_day_volatility'] = data['volatility_(7_days)'].shift(-1)
            data = data.dropna(subset=['next_day_close', 'next_day_volatility'])

            # Extract relevant columns
            features = data[['normalized_vws', 'normalized_volatility', 'normalized_ewma_volatility']].values
            event_flags = data[['event_flag']].values
            targets = data[['next_day_close', 'next_day_volatility']].values

            # Scale features
            features = scaler.fit_transform(features)

            # Separate scalers for each target variable
            target_scalers = [MinMaxScaler() for _ in range(targets.shape[1])]
            scaled_targets = np.hstack([
                target_scalers[i].fit_transform(targets[:, i].reshape(-1, 1))
                for i in range(targets.shape[1])
            ])
            
            features_list.append(torch.tensor(features, dtype=torch.float32))
            event_flags_list.append(torch.tensor(event_flags, dtype=torch.float32))
            targets_list.append(torch.tensor(scaled_targets, dtype=torch.float32))
            lengths_list.append(features.shape[0])  # Sequence length

    # Pad sequences for batch processing
    X_padded = pad_sequence(features_list, batch_first=True)
    event_flags_padded = pad_sequence(event_flags_list, batch_first=True)
    targets_padded = pad_sequence(targets_list, batch_first=True)
    lengths_tensor = torch.tensor(lengths_list, dtype=torch.long)
    
    return X_padded, event_flags_padded, targets_padded, lengths_tensor


def train_model(data_dir, input_dim=3, hidden_dim=64, output_dim=2, dk=64, epochs=100, learning_rate=0.001):
    # Load data
    X_padded, event_flags_padded, targets_padded, lengths_tensor = load_data(data_dir)

    # Initialize the model, loss, and optimizer
    model = StockPredictionModel(input_dim, hidden_dim, output_dim, dk)
    criterion = nn.SmoothL1Loss()  # Huber Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training Loop
    for epoch in range(epochs):
        model.train()
        
        optimizer.zero_grad()
        predictions, attention_weights = model(X_padded, event_flags_padded, lengths_tensor)
        
        # Only take predictions corresponding to valid lengths
        valid_predictions = predictions[range(len(lengths_tensor)), lengths_tensor - 1]
        valid_targets = targets_padded[range(len(lengths_tensor)), lengths_tensor - 1]
        
        loss = criterion(valid_predictions, valid_targets)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
    
    # Compute overall metrics from the attention layer
    attention_layer = model.attention_layer
    avg_bidirectional_mean, avg_bidirectional_std, avg_shape_ratio = attention_layer.compute_overall_metrics()
    print("\nOverall Metrics:")
    print(f"Average Bidirectional Scores Mean: {avg_bidirectional_mean}")
    print(f"Average Bidirectional Scores Std: {avg_bidirectional_std}")
    print(f"Average Shape Ratio: {avg_shape_ratio}")

    evaluate_model(model, X_padded, event_flags_padded, targets_padded, lengths_tensor)

def evaluate_model(model, X_padded, event_flags_padded, targets_padded, lengths_tensor):
    model.eval()
    with torch.no_grad():
        predictions, _ = model(X_padded, event_flags_padded, lengths_tensor)
        valid_predictions = predictions[range(len(lengths_tensor)), lengths_tensor - 1]
        valid_targets = targets_padded[range(len(lengths_tensor)), lengths_tensor - 1]

        # Overall RMSE
        rmse = torch.sqrt(F.mse_loss(valid_predictions, valid_targets))
        print(f"Overall RMSE: {rmse.item()}")

        # Separate flagged and non-flagged events
        flagged_indices = event_flags_padded[range(len(lengths_tensor)), lengths_tensor - 1].squeeze(-1) > 0
        non_flagged_indices = ~flagged_indices

        # Filter predictions and targets
        flagged_predictions = valid_predictions[flagged_indices]
        flagged_targets = valid_targets[flagged_indices]
        non_flagged_predictions = valid_predictions[non_flagged_indices]
        non_flagged_targets = valid_targets[non_flagged_indices]

        # RMSE for flagged events
        if flagged_predictions.numel() > 0:  # Ensure non-empty tensor
            flagged_rmse = torch.sqrt(F.mse_loss(flagged_predictions, flagged_targets))
            print(f"Flagged Event RMSE: {flagged_rmse.item()}")
        else:
            print("No flagged events to compute RMSE.")

        # RMSE for non-flagged events
        if non_flagged_predictions.numel() > 0:  # Ensure non-empty tensor
            non_flagged_rmse = torch.sqrt(F.mse_loss(non_flagged_predictions, non_flagged_targets))
            print(f"Non-Flagged Event RMSE: {non_flagged_rmse.item()}")
        else:
            print("No non-flagged events to compute RMSE.")

        # MAPE (Avoid division by near-zero targets)
        nonzero_mask = valid_targets.abs() > 1e-6  # Exclude near-zero targets
        valid_targets_nonzero = valid_targets[nonzero_mask]
        valid_predictions_nonzero = valid_predictions[nonzero_mask]

        if valid_targets_nonzero.numel() > 0:
            mape = torch.mean(torch.abs((valid_predictions_nonzero - valid_targets_nonzero) / valid_targets_nonzero)) * 100
            print(f"MAPE: {mape.item()}%")
        else:
            print("MAPE: Not computed due to insufficient nonzero target values.")


# Main Execution
if __name__ == "__main__":
    data_dir = "/Users/abhishekjoshi/Documents/GitHub/stock_forecasting_CAI/Data 2"
    train_model(data_dir)
