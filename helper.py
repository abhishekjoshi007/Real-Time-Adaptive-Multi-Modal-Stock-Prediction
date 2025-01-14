import matplotlib.pyplot as plt
import networkx as nx

# Create the graph
G = nx.DiGraph()

# Define nodes for the workflow
nodes = {
    "Input Data": "Input Data",
    "Historical Data": "Historical Data",
    "Sentiment Data": "Sentiment Data",
    "Events Data": "Events Data",
    "Shareholders Data": "Shareholders Data",
    "Company Description": "Company Description",
    "Preprocessing & Feature Engineering": "Preprocessing & Feature Engineering",
    "Feature Extraction": "Feature Extraction",
    "Stock Relation Graph": "Stock Relation Graph",
    "GNN Processing": "Graph Neural Network",
    "Temporal Model": "Temporal Model",
    "AMSPF 1": "AMSPF 1",
    "AMSPF 2": "AMSPF 2",
    "Prediction": "Stock Prediction",
    "Evaluation": "Evaluation Metrics",
    "Output": "Final Recommendations",
}

# Add nodes to the graph
for key, label in nodes.items():
    G.add_node(key, label=label)

# Add edges to define the workflow
edges = [
    ("Input Data", "Historical Data"),
    ("Input Data", "Sentiment Data"),
    ("Input Data", "Events Data"),
    ("Input Data", "Shareholders Data"),
    ("Input Data", "Company Description"),
    ("Historical Data", "Preprocessing & Feature Engineering"),
    ("Sentiment Data", "Preprocessing & Feature Engineering"),
    ("Events Data", "Preprocessing & Feature Engineering"),
    ("Shareholders Data", "Preprocessing & Feature Engineering"),
    ("Company Description", "Preprocessing & Feature Engineering"),
    ("Preprocessing & Feature Engineering", "Feature Extraction"),
    ("Feature Extraction", "Stock Relation Graph"),
    ("Stock Relation Graph", "GNN Processing"),
    ("GNN Processing", "Temporal Model"),
    ("Temporal Model", "AMSPF 1"),
    ("Temporal Model", "AMSPF 2"),
    ("AMSPF 1", "Prediction"),
    ("AMSPF 2", "Prediction"),
    ("Prediction", "Evaluation"),
    ("Evaluation", "Output"),
]
G.add_edges_from(edges)

# Define node positions for a better layout
pos = {
    "Input Data": (0, 5),
    "Historical Data": (2, 6),
    "Sentiment Data": (2, 5.5),
    "Events Data": (2, 5),
    "Shareholders Data": (2, 4.5),
    "Company Description": (2, 4),
    "Preprocessing & Feature Engineering": (4, 5),
    "Feature Extraction": (6, 5),
    "Stock Relation Graph": (8, 5),
    "GNN Processing": (10, 5),
    "Temporal Model": (12, 5),
    "AMSPF 1": (14, 6),
    "AMSPF 2": (14, 4),
    "Prediction": (16, 5),
    "Evaluation": (18, 5),
    "Output": (20, 5),
}

# Draw the graph
plt.figure(figsize=(16, 8))
nx.draw(
    G,
    pos,
    with_labels=True,
    labels=nx.get_node_attributes(G, 'label'),
    node_size=3000,
    node_color="lightblue",
    font_size=10,
    font_weight="bold",
    arrowsize=15,
    edge_color="gray",
)
plt.title("Workflow of AMSPF 1 and AMSPF 2", fontsize=14)
plt.show()
