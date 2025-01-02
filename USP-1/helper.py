import pandas as pd

# Load your dataset
data = pd.read_csv("/Users/abhishekjoshi/Documents/GitHub/stock_forecasting_CAI/Data/AAOI/AAOI_features.csv")

# Compute thresholds
mean_volatility = data['EWMA Volatility'].mean()
std_volatility = data['EWMA Volatility'].std()
k = 1  # Adjust sensitivity if required

high_threshold = mean_volatility + k * std_volatility
low_threshold = mean_volatility - k * std_volatility

# Assign Volatility Class
def assign_volatility_class(vol):
    if vol > high_threshold:
        return "High"
    elif vol < low_threshold:
        return "Low"
    else:
        return "Medium"

data['Recomputed Volatility Class'] = data['EWMA Volatility'].apply(assign_volatility_class)

# Compare classes
discrepancies = data[data['Volatility Class'] != data['Recomputed Volatility Class']]
if not discrepancies.empty:
    print("Discrepancies Found:")
    print(discrepancies)
else:
    print("Volatility Class Assignments Verified.")
