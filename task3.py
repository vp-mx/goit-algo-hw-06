import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

from data import locations, transport_routes_dst


# Create a graph to model the road network of the Odessa region.
G = nx.Graph()
G.add_nodes_from(locations)
G.add_weighted_edges_from(transport_routes_dst)

# Visualize the Graph
plt.figure(figsize=(10, 8))
# Use a spring layout for better spacing between nodes.
pos = nx.spring_layout(G)
nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=3000,
    node_color="lightgreen",
    font_size=12,
    font_weight="bold",
    edge_color="grey",
)
# Get the weights for each edge and draw them on the graph.
edge_labels = nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title("Roads Network of Odessa region with Weights")
plt.show()


# Count the number of nodes and edges.
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()

# Compute the degree (number of connections) for each node.
degrees = dict(G.degree())
degree_df = pd.DataFrame(list(degrees.items()), columns=["Location", "Degree"]).set_index("Location")

print(f"Number of nodes: {num_nodes}")
print(f"Number of edges: {num_edges}")
print(degree_df)


def dijkstra(graph, start):
    """Compute the shortest distance from the start node to every other node
    in the graph using Dijkstra's algorithm.

    :param graph: (dict): Graph represented as a dictionary of dictionaries.
    :param start: (str): The starting node.
    :return: (dict, dict): Tuple of two dictionaries, the first one containing
                            the shortest distances from the start node to each node,
                            and the second one containing the previous node for each node.
    """
    # Initialize distances: set all to infinity except the start node.
    distances = {vertex: float("infinity") for vertex in graph}
    previous = {vertex: None for vertex in graph}
    distances[start] = 0

    # Create a set of all unvisited nodes.
    unvisited = set(graph.keys())

    while unvisited:
        # Choose the unvisited node with the smallest distance.
        current_vertex = min(unvisited, key=lambda vertex: distances[vertex])

        # If the smallest distance is infinity, the rest are unreachable.
        if distances[current_vertex] == float("infinity"):
            break

        # Check each neighbor of the current node.
        for neighbor, attrs in graph[current_vertex].items():
            distance = distances[current_vertex] + attrs["weight"]
            # If a shorter path to neighbor is found, update the distance and record the path.
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current_vertex

        # Mark the current node as visited.
        unvisited.remove(current_vertex)

    return distances, previous


def get_path(previous, start, goal):
    """
    Recover the shortest path from the start node to the goal node using
    the 'previous' dictionary built by the Dijkstra algorithm.

    :param previous: (dict): Dictionary with the previous node for each node.
    :param start: (str): The starting node.
    :param goal: (str): The goal node.
    :return: (list): The list of nodes that form the shortest path.
    """
    path = []
    current = goal
    # Walk backwards from the goal to the start.
    while current is not None:
        path.append(current)
        if current == start:
            break
        current = previous[current]
    path.reverse()

    # Check if a valid path was found.
    if path and path[0] == start:
        return path
    else:
        return None


# Define start and end nodes for the shortest path search.
start_node = "Арциз"
end_node = "Подільськ"

# Convert the graph into a dictionary for our Dijkstra function.
graph_dict = nx.to_dict_of_dicts(G)

# Run Dijkstra's algorithm to get distances and the previous nodes.
distances, previous = dijkstra(graph_dict, start_node)
print(f"\nShortest distance from '{start_node}' to '{end_node}': {distances[end_node]}")

# Recover and print the full shortest path.
path = get_path(previous, start_node, end_node)
print(f"Shortest path: {' -> '.join(path)}" if path else "No path found.")

# Compute Shortest Paths for All Node Pairs
all_pairs_shortest_paths = {}
for node in G.nodes:
    dist, _ = dijkstra(graph_dict, node)
    all_pairs_shortest_paths[node] = dist

print("\nShortest distances between all pairs of nodes:")
for start in all_pairs_shortest_paths:
    for end in all_pairs_shortest_paths[start]:
        print(f"{start} -> {end}: {all_pairs_shortest_paths[start][end]}")
