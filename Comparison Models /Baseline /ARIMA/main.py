import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr

# Define the path to your data folder
data_folder = "/Users/abhishekjoshi/Documents/GitHub/stock_forecasting_CAI/merged_data_usp1_usp3"  # Update with your folder path

# Function to calculate individual metrics
def calculate_metrics(true_values, predicted_values):
    mae = mean_absolute_error(true_values, predicted_values)
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    mape = np.mean(np.abs((true_values - predicted_values) / true_values)) * 100
    directional_accuracy = np.mean(
        np.sign(np.diff(true_values)) == np.sign(np.diff(predicted_values))
    )
    hit_rate = np.sum(np.sign(true_values - predicted_values) == 0) / len(true_values)
    returns = true_values.pct_change().dropna()
    sharpe_ratio = returns.mean() / returns.std() if returns.std() != 0 else 0
    ic, _ = spearmanr(true_values, predicted_values)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape,
        "Directional Accuracy (%)": directional_accuracy * 100,
        "Hit Rate (%)": hit_rate * 100,
        "Sharpe Ratio": sharpe_ratio,
        "Information Coefficient (IC)": ic,
    }

# Initialize accumulators for combined metrics
overall_metrics = {
    "MAE": [],
    "RMSE": [],
    "MAPE (%)": [],
    "Directional Accuracy (%)": [],
    "Hit Rate (%)": [],
    "Sharpe Ratio": [],
    "Information Coefficient (IC)": []
}

# Process each file in the data folder
for file_name in os.listdir(data_folder):
    if file_name.endswith(".csv"):  # Ensure we process only CSV files
        file_path = os.path.join(data_folder, file_name)

        # Load the data
        data = pd.read_csv(file_path)

        # Ensure 'Close' column exists
        if 'Close' not in data.columns:
            print(f"File {file_name} does not contain 'Close' column. Skipping.")
            continue

        # Sort data by date
        data['Date'] = pd.to_datetime(data['Date'])
        data.sort_values('Date', inplace=True)

        # Generate random walk predictions
        data['Random_Walk'] = data['Close'].shift(1)

        # Drop rows with NaN values in 'Close' or 'Random_Walk'
        data.dropna(subset=['Close', 'Random_Walk'], inplace=True)

        # Calculate metrics for the current file if sufficient data exists
        if len(data) > 1:  # Ensure there is enough data to calculate metrics
            metrics = calculate_metrics(data['Close'], data['Random_Walk'])

            # Accumulate metrics for combined overall values
            for key in overall_metrics.keys():
                overall_metrics[key].append(metrics[key])
        else:
            print(f"Not enough data in {file_name} after processing. Skipping.")

# Calculate overall combined metrics
combined_metrics = {key: np.mean(values) for key, values in overall_metrics.items()}

# Display the overall combined metrics
print("Combined Overall Metrics:")
for metric, value in combined_metrics.items():
    print(f"{metric}: {value:.4f}")
