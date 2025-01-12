import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os

# Load and preprocess the data
def load_data(data_dir):
    feature_data = []
    target_data = []
    event_flags = []
    
    for ticker_folder in os.listdir(data_dir):
        ticker_path = os.path.join(data_dir, ticker_folder)
        input_file = os.path.join(ticker_path, f"{ticker_folder}_usp3_prepared_data.csv")
        
        if os.path.exists(input_file):
            # Load processed data
            data = pd.read_csv(input_file)
            data.columns = data.columns.str.replace(' ', '_').str.lower()  # Standardize column names
            
            # Create next-day close and volatility as target variables
            data['next_day_close'] = data['close'].shift(-1)
            data['next_day_volatility'] = data['volatility_(7_days)'].shift(-1)
            data = data.dropna(subset=['next_day_close', 'next_day_volatility'])
            
            # Extract features and targets
            features = data[['normalized_vws', 'normalized_volatility', 'normalized_ewma_volatility', 'event_flag']]
            targets = data[['next_day_close', 'next_day_volatility']]
            
            feature_data.append(features)
            target_data.append(targets)
            event_flags.append(features['event_flag'])
    
    # Concatenate all data
    feature_data = pd.concat(feature_data, axis=0)
    target_data = pd.concat(target_data, axis=0)
    event_flags = pd.concat(event_flags, axis=0)
    
    return feature_data, target_data, event_flags

# Calculate additional metrics
def calculate_metrics(predictions, actuals, returns, event_flags, risk_free_rate=0.01):
    metrics = {}
    
    # RMSE and MAE
    metrics['RMSE'] = np.sqrt(mean_squared_error(actuals, predictions))
    metrics['MAE'] = mean_absolute_error(actuals, predictions)
    
    # Bidirectional Value
    metrics['Bidirectional_Value'] = np.mean(predictions - actuals)
    
    # Information Coefficient (IC)
    ic = np.corrcoef(predictions.flatten(), actuals.flatten())[0, 1]
    metrics['IC'] = ic if not np.isnan(ic) else 0.0
    
    # Sharpe Ratio
    excess_returns = returns - risk_free_rate
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)
    metrics['Sharpe_Ratio'] = sharpe_ratio if not np.isnan(sharpe_ratio) else 0.0
    
    # MAPE
    nonzero_mask = actuals != 0  # Avoid division by zero
    mape = np.mean(np.abs((predictions[nonzero_mask] - actuals[nonzero_mask]) / actuals[nonzero_mask])) * 100
    metrics['MAPE'] = mape if not np.isnan(mape) else 0.0
    
    # Directional Accuracy for Event-Flagged Days
    flagged_indices = event_flags == 1
    if flagged_indices.sum() > 0:
        flagged_predictions = predictions[flagged_indices]
        flagged_actuals = actuals[flagged_indices]
        
        # Use slicing to mimic the shift operation
        shifted_actuals = np.roll(flagged_actuals, -1)  # Shift actual values by -1
        shifted_actuals[-1] = flagged_actuals[-1]  # Avoid index mismatch for the last element
        
        correct_directions = np.sum(
            (flagged_predictions - flagged_actuals) * (shifted_actuals - flagged_actuals) > 0
        )
        metrics['Directional_Accuracy_Flagged'] = correct_directions / len(flagged_predictions)
    else:
        metrics['Directional_Accuracy_Flagged'] = np.nan  # No flagged events
    
    # Return the metrics dictionary
    return metrics

# Train and evaluate the linear regression model
def run_linear_regression(data_dir):
    # Load data
    features, targets, event_flags = load_data(data_dir)
    
    # Feature Engineering: Add interaction terms
    features['vws_x_volatility'] = features['normalized_vws'] * features['normalized_volatility']
    features['ewma_x_event'] = features['normalized_ewma_volatility'] * features['event_flag']
    
    # Scale the features and targets
    feature_scaler = MinMaxScaler()
    scaled_features = feature_scaler.fit_transform(features)
    
    target_scalers = [MinMaxScaler(), MinMaxScaler()]
    scaled_targets = np.hstack([
        target_scalers[i].fit_transform(targets.iloc[:, i].values.reshape(-1, 1))
        for i in range(targets.shape[1])
    ])
    
    # Train-test split
    X_train, X_test, y_train, y_test, flags_train, flags_test = train_test_split(
        scaled_features, scaled_targets, event_flags, test_size=0.2, random_state=42
    )
    
    # Train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Calculate metrics
    returns = predictions[:, 0] - y_test[:, 0]  # Return = predicted close - actual close
    metrics = calculate_metrics(predictions, y_test, returns, flags_test)
    
    # Print metrics
    print("Enhanced Linear Regression Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save sample output
    sample_output = pd.DataFrame({
        'Predicted_Close': predictions[:, 0],
        'Actual_Close': y_test[:, 0],
        'Predicted_Volatility': predictions[:, 1],
        'Actual_Volatility': y_test[:, 1],
        'Event_Flag': flags_test.values
    })
    sample_output.to_csv("enhanced_linear_regression_output.csv", index=False)
    print("Sample output saved to enhanced_linear_regression_output.csv")

# Main execution
if __name__ == "__main__":
    data_dir = "/Users/abhishekjoshi/Documents/GitHub/stock_forecasting_CAI/USP 3 data"  # Update this path
    run_linear_regression(data_dir)
