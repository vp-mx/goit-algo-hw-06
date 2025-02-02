import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

from data import locations, transport_routes

G = nx.Graph()
G.add_nodes_from(locations)
G.add_edges_from(transport_routes)
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=12, font_weight="bold")
plt.show()

num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
degrees = dict(G.degree())
degree_df = pd.DataFrame(degrees.items(), columns=["Location", "Degree"])

print(f"Number of nodes: {num_nodes}")
print(f"Number of edges: {num_edges}")
print(degree_df)


# DFS & BFS
def dfs_path(graph, start, goal):
    """Find all paths from the start node to the goal node using Depth-First Search (DFS).

    :param graph: A NetworkX graph object.
    :param start: The starting node.
    :param goal: The goal node.
    :yield: A list representing a path from start to goal.
    """
    # Initialize the stack with a tuple containing the start node and the path so far.
    stack = [(start, [start])]

    # Loop until there are no more nodes to explore.
    while stack:
        # Remove the last element from the stack (LIFO order).
        (vertex, path) = stack.pop()

        # Get all neighbors of the current node that are not already in the path.
        for next in set(graph.neighbors(vertex)) - set(path):
            # If the neighbor is the goal, yield the complete path.
            if next == goal:
                yield path + [next]
            else:
                # Otherwise, add the neighbor and the new path to the stack.
                stack.append((next, path + [next]))


def bfs_path(graph, start, goal):
    """
    Find all paths from the start node to the goal node using Breadth-First Search (BFS).

    :param graph: A NetworkX graph object.
    :param start: The starting node.
    :param goal: The goal node.
    :yield: A list representing a path from start to goal.
    """
    queue = [(start, [start])]

    # Loop until the queue is empty.
    while queue:
        # Remove the element from the left side.
        (vertex, path) = queue.pop(0)

        # Get all neighbors of the current node that are not already in the path.
        for next in set(graph.neighbors(vertex)) - set(path):
            # If the neighbor is the goal, yield the complete path.
            if next == goal:
                yield path + [next]
            else:
                # Otherwise, add the neighbor and the new path to the end of the queue.
                queue.append((next, path + [next]))


start_node, end_node = "Болград", "Южне"
dfs_paths = list(dfs_path(G, start_node, end_node))
bfs_paths = list(bfs_path(G, start_node, end_node))

print("\nDFS Paths:")
for path in dfs_paths:
    print(path)

print("\nBFS Paths:")
for path in bfs_paths:
    print(path)

# Results comparison
print(f"\nNumber of DFS Paths: {len(dfs_paths)}")
print(f"Number of BFS Paths: {len(bfs_paths)}")
