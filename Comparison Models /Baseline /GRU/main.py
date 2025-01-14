import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

# Load and preprocess data function (same as before)
def load_and_preprocess_data(ticker, base_path, sequence_length=5):
    file_path = os.path.join(base_path, f"{ticker}.csv")
    data = pd.read_csv(file_path)
    prices = data['Close'].values.reshape(-1, 1)
    
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    prices_scaled = scaler.fit_transform(prices)
    
    X, y = [], []
    for i in range(sequence_length, len(prices_scaled)):
        X.append(prices_scaled[i-sequence_length:i, 0])
        y.append(prices_scaled[i, 0])
    
    X = np.array(X)
    y = np.array(y)
    
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y, scaler

# Build GRU model function
def build_gru_model(input_shape, units=50, dropout_rate=0.2):
    model = Sequential()
    
    # GRU layers
    model.add(GRU(units=units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(GRU(units=units, return_sequences=False))
    model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Training and evaluating the GRU model function
def train_and_evaluate_gru(ticker, base_path, output_path, sequence_length=5, epochs=10, batch_size=32):
    # Load and preprocess data
    X, y, scaler = load_and_preprocess_data(ticker, base_path, sequence_length)
    
    # Split data into training and testing sets
    split_index = int(len(X) * 0.8)  # 80% training, 20% testing
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    # Build the model
    model = build_gru_model(X_train.shape[1:])
    
    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Inverse scale predictions and actual values
    predictions_rescaled = scaler.inverse_transform(predictions)
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Save predictions to a file
    result_df = pd.DataFrame({
        'Actual': y_test_rescaled.flatten(),
        'Predicted': predictions_rescaled.flatten()
    })
    result_df.to_csv(os.path.join(output_path, f"{ticker}_gru_predictions.csv"), index=False)
    
    # Evaluate the model
    mse = mean_squared_error(y_test_rescaled, predictions_rescaled)
    mae = mean_absolute_error(y_test_rescaled, predictions_rescaled)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test_rescaled - predictions_rescaled) / y_test_rescaled)) * 100
    
    # Directional Accuracy
    actual_diff = np.diff(y_test_rescaled.flatten())
    predicted_diff = np.diff(predictions_rescaled.flatten())
    directional_accuracy = np.mean(np.sign(actual_diff) == np.sign(predicted_diff)) * 100
    
    # Hit Rate (Predicted direction matching actual direction)
    hit_rate = np.mean((np.sign(actual_diff) == np.sign(predicted_diff)) & (np.sign(predicted_diff) != 0)) * 100
    
    # Sharpe Ratio (Assuming returns are the daily percentage change of predictions)
    daily_returns = np.diff(predictions_rescaled.flatten()) / predictions_rescaled[:-1].flatten()
    sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns)
    
    # Information Coefficient (IC) - Correlation between predicted and actual returns
    actual_returns = np.diff(y_test_rescaled.flatten()) / y_test_rescaled[:-1].flatten()
    information_coefficient = np.corrcoef(predicted_diff, actual_diff)[0, 1]
    
    print(f"""
    Ticker: {ticker}
    MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}%
    Directional Accuracy: {directional_accuracy:.4f}%, Hit Rate: {hit_rate:.4f}%
    Sharpe Ratio: {sharpe_ratio:.4f}, Information Coefficient: {information_coefficient:.4f}
    """)
    
    return mse, mae, rmse, mape, directional_accuracy, hit_rate, sharpe_ratio, information_coefficient

import os

# Define paths
data_folder = "/Users/gurojaschadha/Downloads/Combined Data /merged_data_usp1_usp3"
output_folder = "/Users/gurojaschadha/Downloads/Predictions"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Initialize results list
results = []

# Iterate through all CSV files in the folder
for file_name in os.listdir(data_folder):
    if file_name.endswith(".csv"):  # Ensure only CSV files are processed
        ticker = file_name.split(".")[0]
        print(f"Processing file: {file_name} for ticker: {ticker}")
        
        try:
            metrics = train_and_evaluate_gru(ticker, data_folder, output_folder)
            results.append({
                'Ticker': ticker,
                'MSE': metrics[0],
                'MAE': metrics[1],
                'RMSE': metrics[2],
                'MAPE': metrics[3],
                'Directional Accuracy': metrics[4],
                'Hit Rate': metrics[5],
                'Sharpe Ratio': metrics[6],
                'Information Coefficient': metrics[7]
            })
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

# Save results summary
results_df = pd.DataFrame(results)
results_summary_path = os.path.join(output_folder, "gru_results_summary.csv")
results_df.to_csv(results_summary_path, index=False)
print(f"Results saved to {results_summary_path}")

# Calculate combined overall metrics
combined_metrics = {
    'MAE': np.mean([result['MAE'] for result in results]),
    'RMSE': np.mean([result['RMSE'] for result in results]),
    'MAPE (%)': np.mean([result['MAPE'] for result in results]),
    'Directional Accuracy (%)': np.mean([result['Directional Accuracy'] for result in results]),
    'Hit Rate (%)': np.mean([result['Hit Rate'] for result in results]),
    'Sharpe Ratio': np.mean([result['Sharpe Ratio'] for result in results]),
    'Information Coefficient (IC)': np.mean([result['Information Coefficient'] for result in results])
}

# Create a DataFrame for combined metrics
combined_metrics_df = pd.DataFrame([combined_metrics])
combined_metrics_df.to_csv(os.path.join(output_folder, "combined_gru_metrics.csv"), index=False)
print(f"Combined metrics saved to {os.path.join(output_folder, 'combined_gru_metrics.csv')}")
