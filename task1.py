import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

from data import transport_routes, locations

# Create a graph to model the road network of the Odessa region.
G = nx.Graph()
G.add_nodes_from(locations)
# Add edges to the graph
G.add_edges_from(transport_routes)
# Visualize the Graph
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=3000, node_color="yellow", font_size=12, font_weight="bold")
plt.show()

# Graph Analysis
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
degrees = dict(G.degree())
degree_df = pd.DataFrame(degrees.items(), columns=["Location", "Degree"])

print(f"Number of nodes: {num_nodes}")
print(f"Number of edges: {num_edges}")
print(degree_df)
