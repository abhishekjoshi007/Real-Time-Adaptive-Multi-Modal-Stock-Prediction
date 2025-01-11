from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import json

# Load and preprocess dataset
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
    return features, labels

# Main script
if __name__ == "__main__":
    tickers = [
        'PD', 'HEAR', 'AAPL', 'PANW', 'ARRY', 'TEL'
    ]

    cumulative_metrics = {
        "MAE": [],
        "RMSE": [],
        "Sharpe Ratio": [],
        "Directional Accuracy": [],
        "IC": []
    }

    for ticker in tickers:
        try:
            print(f"Processing ticker: {ticker}")
            file_path = f"/Users/abhishekjoshi/Documents/GitHub/stock_forecasting_CAI/USP 1 Data/{ticker}/{ticker}_merged_with_vix.csv"
            features, labels = load_data(file_path)

            # Initialize Linear Regression model
            model = LinearRegression()

            # Train the model
            model.fit(features, labels)

            # Predict the outputs
            predictions = model.predict(features)

            # Calculate metrics
            mae = mean_absolute_error(labels, predictions)
            rmse = np.sqrt(mean_squared_error(labels, predictions))
            sharpe_ratio = np.mean(predictions) / np.std(predictions)
            directional_accuracy = np.mean(
                np.sign(predictions) == np.sign(labels)
            ) * 100
            ic, _ = pearsonr(predictions, labels)

            # Add metrics to cumulative dictionary
            cumulative_metrics["MAE"].append(mae)
            cumulative_metrics["RMSE"].append(rmse)
            cumulative_metrics["Sharpe Ratio"].append(sharpe_ratio)
            cumulative_metrics["Directional Accuracy"].append(directional_accuracy)
            cumulative_metrics["IC"].append(ic)

            # Save individual results
            results = {
                "Ticker": ticker,
                "MAE": float(mae),
                "RMSE": float(rmse),
                "Sharpe Ratio": float(sharpe_ratio),
                "Directional Accuracy": float(directional_accuracy),
                "IC": float(ic)
            }

            output_file = f"/Users/abhishekjoshi/Documents/GitHub/stock_forecasting_CAI/USP 1 Data/{ticker}/evaluation_results_linear_{ticker}.json"
            with open(output_file, "w") as f:
                json.dump(results, f)

            print(f"Linear Regression evaluation results saved for {ticker} in {output_file}.")

        except Exception as e:
            print(f"Error processing ticker {ticker}: {e}")

    # Calculate cumulative metrics
    print("\nCalculating cumulative evaluation metrics...")
    cumulative_results = {
        "Average MAE": float(np.mean(cumulative_metrics["MAE"])),
        "Average RMSE": float(np.mean(cumulative_metrics["RMSE"])),
        "Average Sharpe Ratio": float(np.mean(cumulative_metrics["Sharpe Ratio"])),
        "Average Directional Accuracy": float(np.mean(cumulative_metrics["Directional Accuracy"])),
        "Average IC": float(np.mean(cumulative_metrics["IC"]))
    }

    # Save cumulative results
    with open("cumulative_evaluation_results_linear.json", "w") as f:
        json.dump(cumulative_results, f)

    print("Cumulative evaluation metrics saved in cumulative_evaluation_results_linear.json.")
