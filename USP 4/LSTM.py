import pandas as pd

# Load your combined output as a DataFrame
data = pd.read_csv('/Users/abhishekjoshi/Documents/GitHub/stock_forecasting_CAI/USP 4/daily_recommendations.csv')  # Replace with your actual file

# Average Return
average_return = data['Close'].mean()
print("Average Return:", average_return)

# Hit Rate
hit_rate = len(data[data['Close'] > 0]) / len(data)
print("Hit Rate:", hit_rate)

# Sharpe Ratio
risk_free_rate = 0.02
excess_returns = data['Close'] - risk_free_rate
sharpe_ratio = excess_returns.mean() / excess_returns.std()
print("Sharpe Ratio:", sharpe_ratio)
