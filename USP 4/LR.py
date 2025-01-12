# Updated Linear Regression Model
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV

# Load Data
data = pd.read_csv("/Users/abhishekjoshi/Documents/GitHub/stock_forecasting_CAI/USP 4/daily_recommendations.csv")

# Feature Selection
features = ["Momentum", "Volatility", "EWMA_Volatility", "Composite_Score", "Sentiment Score"]
target = "Close"

X = data[features]
y = data[target]

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Ridge Regression with Hyperparameter Tuning
ridge = Ridge()
param_grid = {
    'alpha': [0.1, 1.0, 10.0, 100.0]  # Regularization strengths
}

# Grid Search for Hyperparameter Tuning
grid_search = GridSearchCV(ridge, param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train, y_train)

# Best Model
best_model = grid_search.best_estimator_

# Predictions
y_pred = best_model.predict(X_test)

# Metrics
# Average Return
average_return = np.mean(y_pred)

# Hit Rate
hit_rate = np.mean((np.sign(y_pred) == np.sign(y_test)).astype(int))

# Sharpe Ratio
excess_returns = y_pred - np.mean(y_pred)
sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) != 0 else 0

# RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Bi-Directional Value
bi_directional = np.corrcoef(y_test, y_pred)[0, 1]  # Correlation as an example metric

# Display Results
print("Best Hyperparameters:", grid_search.best_params_)
print("Average Return:", average_return)
print("Hit Rate:", hit_rate * 100, "%")
print("Sharpe Ratio:", sharpe_ratio)
print("RMSE:", rmse)
print("Bi-Directional Value:", bi_directional)
