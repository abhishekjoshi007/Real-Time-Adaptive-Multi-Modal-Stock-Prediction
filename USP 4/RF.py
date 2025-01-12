#Random Forest
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np

# Load Data
data = pd.read_csv("/Users/abhishekjoshi/Documents/GitHub/stock_forecasting_CAI/USP 4/daily_recommendations.csv")

# Feature Selection
features = ["Momentum", "Volatility", "Composite_Score"]
target = "Close"

# Train-Test Split
train_size = int(0.8 * len(data))
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]

X_train, y_train = train_data[features], train_data[target]
X_test, y_test = test_data[features], test_data[target]

# Train Random Forest Model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics Calculation
average_return = np.mean(y_pred)
hit_rate = np.mean((y_pred > 0) == (y_test > 0))
sharpe_ratio = average_return / np.std(y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
bi_directional_value = accuracy_score((y_test > 0).astype(int), (y_pred > 0).astype(int))

# Results
print(f"Average Return: {average_return}")
print(f"Hit Rate: {hit_rate}")
print(f"Sharpe Ratio: {sharpe_ratio}")
print(f"RMSE: {rmse}")
print(f"Bi-Directional Value: {bi_directional_value}")
