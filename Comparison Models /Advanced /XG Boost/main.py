import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# Define the path to your data folder
data_folder = "/Users/abhishekjoshi/Documents/GitHub/stock_forecasting_CAI/merged_data_usp1_usp3"

# Function to calculate metrics
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
results = []
for file_name in os.listdir(data_folder):
    if file_name.endswith(".csv"):  # Ensure we process only CSV files
        file_path = os.path.join(data_folder, file_name)
        ticker = file_name.split(".")[0]  # Extract ticker name from the file name

        # Load the data
        data = pd.read_csv(file_path)

        # Ensure 'Close' column exists
        if 'Close' not in data.columns:
            print(f"File {file_name} does not contain 'Close' column. Skipping.")
            continue

        # Sort data by date
        data['Date'] = pd.to_datetime(data['Date'])
        data.sort_values('Date', inplace=True)

        # Feature engineering
        features = data.drop(columns=['Date', 'Close', 'Ticker Name'])  # Exclude non-numeric columns
        target = data['Close']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42, shuffle=False)

        # Ensure sufficient training and test data
        if len(X_train) < 10 or len(X_test) < 10:
            print(f"Not enough data in {file_name} for training. Skipping.")
            continue

        # Train XGBoost model
        model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        # Calculate metrics for the current file
        metrics = calculate_metrics(y_test, predictions)
        metrics['Ticker'] = ticker  # Add ticker name to the results
        results.append(metrics)

        # Accumulate metrics for combined overall values
        for key in overall_metrics.keys():
            overall_metrics[key].append(metrics[key])

# Calculate overall combined metrics
combined_metrics = {key: np.mean(values) for key, values in overall_metrics.items()}

# Display per-ticker metrics and combined metrics
results_df = pd.DataFrame(results)
print("\nPer-Ticker Metrics:")
print(results_df)

print("\nCombined Overall Metrics:")
for metric, value in combined_metrics.items():
    print(f"{metric}: {value:.4f}")
