#Linear Regression Model
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Path to the GML file
gml_file = "stock_graph_with_edges.gml"

# Step 1: Load the Graph from GML File
def load_graph_from_gml(gml_file):
    graph = nx.read_gml(gml_file)
    features = []
    labels = []

    # Extract node features and labels
    for node, data in graph.nodes(data=True):
        # Derived features
        volume_weighted_sentiment = data.get('volume_weighted_sentiment', 0)
        daily_return = data.get('daily_return', 0)
        rolling_avg = data.get('rolling_avg', 0)
        volatility = data.get('volatility', 0)
        momentum = data.get('momentum', 0)

        # Add derived features
        features.append([
            volume_weighted_sentiment,
            daily_return,
            rolling_avg,
            volatility,
            momentum,
            daily_return ** 2,  # Derived feature
            rolling_avg * volatility,  # Interaction term
        ])

        # Label
        target = data.get('daily_return', None)
        if target is None:
            print(f"Warning: Node {data.get('label')} has missing target. Defaulting to 0.")
            target = 0
        labels.append(target)

    return np.array(features), np.array(labels)

# Step 2: Preprocess Features
def preprocess_features(features):
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features)
    return normalized_features

# Step 3: Train/Test Split
def split_data(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, shuffle=True
    )
    return X_train, X_test, y_train, y_test

# Step 4: Evaluate Metrics
def evaluate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        ic = 0  # Set IC to 0 if variance is zero
    else:
        ic = np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]
    directional_accuracy = (np.sign(y_true) == np.sign(y_pred)).sum() / len(y_true)
    return mae, rmse, ic, directional_accuracy

# Step 5: Save Results to File
def save_results(mae, rmse, ic, directional_accuracy, output_file="baseline_results.txt"):
    with open(output_file, "w") as file:
        file.write("Linear Regression Model - Evaluation Metrics:\n")
        file.write(f"Train RMSE: {rmse:.6f}, Train MAE: {mae:.6f}\n")
        file.write(f"Information Coefficient (IC): {ic:.6f}\n")
        file.write(f"Directional Accuracy: {directional_accuracy:.6f}\n")
    print(f"Results saved to {output_file}")

# Main Execution
def main():
    # Load graph and preprocess data
    features, labels = load_graph_from_gml(gml_file)
    features = preprocess_features(features)

    print("Features Sample:", features[:5])
    print("Labels Sample:", labels[:5])
    print("Feature Variance:", np.var(features, axis=0))
    print("Label Variance:", np.var(labels))

    # Split data
    X_train, X_test, y_train, y_test = split_data(features, labels)

    # Initialize Ridge regression model
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Evaluate Metrics
    train_mae, train_rmse, train_ic, train_directional_accuracy = evaluate_metrics(y_train, y_train_pred)
    test_mae, test_rmse, test_ic, test_directional_accuracy = evaluate_metrics(y_test, y_test_pred)

    # Output results
    print("Linear Regression Model - Evaluation Metrics:")
    print(f"Train RMSE: {train_rmse:.6f}, Train MAE: {train_mae:.6f}")
    print(f"Test RMSE: {test_rmse:.6f}, Test MAE: {test_mae:.6f}")
    print(f"Information Coefficient (IC): {test_ic:.6f}")
    print(f"Directional Accuracy: {test_directional_accuracy:.6f}")

    # Save results
    save_results(test_mae, test_rmse, test_ic, test_directional_accuracy)

if __name__ == "__main__":
    main()
