import os
import pandas as pd
import torch

# Directory containing your processed files
data_dir = "/Users/abhishekjoshi/Documents/GitHub/stock_forecasting_CAI/Data 2"

# Containers to hold data
all_X = []  # All features
all_event_flags = []  # All Event Flags
lengths = []  # Sequence lengths for each ticker

# Loop through all ticker folders
for ticker_folder in os.listdir(data_dir):
    ticker_path = os.path.join(data_dir, ticker_folder)
    input_file = os.path.join(ticker_path, f"{ticker_folder}_usp3_prepared_data.csv")
    
    if os.path.exists(input_file):
        # Load processed data
        data = pd.read_csv(input_file)
        
        # Extract relevant columns
        features = data[['Normalized VWS', 'Normalized Volatility', 'Normalized EWMA Volatility']].values
        event_flags = data[['Event Flag']].values

        # Add to containers
        all_X.append(torch.tensor(features, dtype=torch.float32))
        all_event_flags.append(torch.tensor(event_flags, dtype=torch.float32))
        lengths.append(len(features))  # Sequence length for this ticker

        print(f"Loaded data for ticker: {ticker_folder}")
    else:
        print(f"File not found for ticker: {ticker_folder}")

# Pad sequences for uniform batch input
X_padded = torch.nn.utils.rnn.pad_sequence(all_X, batch_first=True)  # Shape: (batch_size, max_seq_len, input_dim)
event_flags_padded = torch.nn.utils.rnn.pad_sequence(all_event_flags, batch_first=True)  # Shape: (batch_size, max_seq_len, 1)
lengths_tensor = torch.tensor(lengths, dtype=torch.int64)  # Store lengths for packing/unpacking

print("Feature Tensor Shape:", X_padded.shape)  # (batch_size, max_seq_len, input_dim)
print("Event Flag Tensor Shape:", event_flags_padded.shape)  # (batch_size, max_seq_len, 1)
print("Lengths:", lengths_tensor)
