import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Define the path to your data folder
data_folder = "/Users/abhishekjoshi/Documents/GitHub/stock_forecasting_CAI/merged_data_usp1_usp3"  # Update with the correct path

# Define the metrics calculation function
def calculate_metrics(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    # Directional accuracy
    actual_direction = np.sign(np.diff(y_test))
    predicted_direction = np.sign(np.diff(y_pred))
    directional_accuracy = np.mean(actual_direction == predicted_direction)

    # Hit rate
    hit_rate = np.mean((y_test > np.mean(y_test)) == (y_pred > np.mean(y_pred)))

    # Sharpe ratio (simplified, using return differences)
    returns = y_test.pct_change().dropna()
    excess_returns = returns - np.mean(returns)
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)

    # Information Coefficient (IC)
    ic = np.corrcoef(y_test, y_pred)[0, 1]

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape,
        "Directional Accuracy (%)": directional_accuracy * 100,
        "Hit Rate (%)": hit_rate * 100,
        "Sharpe Ratio": sharpe_ratio,
        "Information Coefficient (IC)": ic,
    }

# Define the features to use
selected_features = [
    'Volatility (7 Days)_USP1_2',  # USP 1 and 2 feature
    'Volume-Weighted Sentiment',  # USP 2 feature
    'Volatility (7 Days)_USP3',   # USP 3 feature
    'Momentum (7 Days)',          # Momentum as additional feature
    'Close'                       # Target variable
]

# Initialize variables for combined metrics
combined_metrics = {
    "MAE": [],
    "RMSE": [],
    "MAPE (%)": [],
    "Directional Accuracy (%)": [],
    "Hit Rate (%)": [],
    "Sharpe Ratio": [],
    "Information Coefficient (IC)": []
}

# Iterate over all CSV files in the data folder
for ticker_file in os.listdir(data_folder):
    if ticker_file.endswith(".csv"):
        ticker_name = os.path.splitext(ticker_file)[0]
        file_path = os.path.join(data_folder, ticker_file)

        # Load the dataset
        data = pd.read_csv(file_path)

        # Check if all required columns exist
        if not all(feature in data.columns for feature in selected_features):
            print(f"Skipping {ticker_name} due to missing features.")
            continue

        # Extract features and target
        X = data[selected_features[:-1]]  # All columns except 'Close'
        y = data['Close']                 # Target variable

        # Handle missing values
        X.fillna(X.mean(), inplace=True)
        y.fillna(y.mean(), inplace=True)

        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train the XGBoost model
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective='reg:squarederror',
            random_state=42
        )

        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics and add to combined results
        metrics = calculate_metrics(y_test, y_pred)
        for key in combined_metrics.keys():
            combined_metrics[key].append(metrics[key])

# Compute aggregated metrics
aggregated_metrics = {key: np.mean(values) for key, values in combined_metrics.items()}

# Save the aggregated metrics to a CSV
aggregated_metrics_df = pd.DataFrame([aggregated_metrics])
aggregated_metrics_df.to_csv("aggregated_xgboost_metrics.csv", index=False)

print("Combined metrics for all tickers saved to 'aggregated_xgboost_metrics.csv'.")
