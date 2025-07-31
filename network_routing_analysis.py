import networkx as nx
import random
import math
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd

# Shared color scheme
algo_colors = {
    'OSPF': 'blue',
    'ACO': 'green',
    'Q-Routing': 'purple',
    'DCR': 'red'
}

# --- Step 1: Ask user for network load ---
print("Select Network Load:")
print("1. Low Load")
print("2. Medium Load")
print("3. High Load")
load_choice = input("Enter choice (1/2/3): ")

# Updated load selection with cleaner mapping
p = {'1': 0.05, '2': 0.15, '3': 0.30}.get(load_choice, 0.15)
load_label = {'1': 'Low Load', '2': 'Medium Load', '3': 'High Load'}.get(load_choice, 'Medium Load')
print(f"\nRunning tests with {load_label} using p = {p}")

# ----------------------------
# Generate Network Topology
# ----------------------------
def generate_topology(n=50, p=0.3, seed=42):
    """Generate a connected Erdős-Rényi graph with weighted edges"""
    random.seed(seed)
    G = nx.erdos_renyi_graph(n, p, directed=False, seed=seed)
    
    # Ensure connectivity
    while not nx.is_connected(G):
        G = nx.erdos_renyi_graph(n, p, directed=False, seed=seed)
    
    # Assign random weights to edges
    for (u, v) in G.edges():
        G[u][v]['weight'] = random.randint(1, 10)
    
    return G

# ----------------------------
# Routing Algorithms
# ----------------------------

def ospf_routing(G, source, target):
    """OSPF routing using Dijkstra's shortest path algorithm"""
    try:
        return nx.shortest_path(G, source=source, target=target, weight='weight')
    except nx.NetworkXNoPath:
        return []

def aco_routing(G, source, target, alpha=1, beta=2, iterations=10, ants=5):
    """Ant Colony Optimization routing algorithm"""
    pheromone = defaultdict(lambda: 1.0)
    best_path = []
    best_cost = float('inf')
    
    for _ in range(iterations):
        for _ in range(ants):
            path = [source]
            visited = set(path)
            current = source
            
            while current != target:
                neighbors = [n for n in G.neighbors(current) if n not in visited]
                if not neighbors: 
                    break
                
                probs = []
                for n in neighbors:
                    tau = pheromone[(current, n)] ** alpha
                    eta = (1 / G[current][n]['weight']) ** beta
                    probs.append(tau * eta)
                
                total = sum(probs)
                if total == 0: 
                    break
                
                probs = [p / total for p in probs]
                next_node = random.choices(neighbors, weights=probs, k=1)[0]
                path.append(next_node)
                visited.add(next_node)
                current = next_node
            
            if current == target and len(path) > 1:
                cost = sum(G[path[i]][path[i+1]]['weight'] for i in range(len(path) - 1))
                if cost < best_cost:
                    best_cost = cost
                    best_path = path
        
        # Update pheromone
        if best_path:
            for i in range(len(best_path) - 1):
                pheromone[(best_path[i], best_path[i+1])] += 1.0 / best_cost
    
    return best_path

def q_routing(G, source, target, gamma=0.9, episodes=100):
    """Q-learning based routing algorithm"""
    Q = defaultdict(lambda: defaultdict(lambda: 0.0))
    
    # Training phase
    for _ in range(episodes):
        current = source
        visited = set()
        
        while current != target and len(visited) < len(G.nodes()):
            visited.add(current)
            neighbors = list(G.neighbors(current))
            if not neighbors: 
                break
            
            next_node = random.choice(neighbors)
            reward = -G[current][next_node]['weight']
            max_q_next = max(Q[next_node].values(), default=0)
            Q[current][next_node] += 0.1 * (reward + gamma * max_q_next - Q[current][next_node])
            current = next_node
    
    # Path extraction
    path = [source]
    current = source
    visited = set(path)
    
    while current != target:
        if not Q[current]: 
            break
        next_node = max(Q[current], key=Q[current].get)
        if next_node in visited: 
            break
        path.append(next_node)
        visited.add(next_node)
        current = next_node
    
    return path if current == target else []

# ----------------------------
# DCR Algorithm
# ----------------------------
weight_vector = [0.3, 0.3, 0.4]

def normalize_factors(route, G):
    """Normalize routing factors for DCR algorithm"""
    if len(route) < 2:
        return [0, 0, 0]
    
    hop_count = len(route) - 1
    path_weight = sum(G[route[i]][route[i+1]]['weight'] for i in range(len(route)-1))
    avg_degree = sum(len(list(G.neighbors(n))) for n in route) / len(route)
    
    # Normalization constants
    max_hop, max_weight, max_degree = 10, 100, 10
    
    return [hop_count / max_hop, path_weight / max_weight, avg_degree / max_degree]

def activation_energy(phi_vector, weight_vector):
    """Calculate activation energy for DCR"""
    return sum(w * phi for w, phi in zip(weight_vector, phi_vector))

def catalytic_activity(Ea, T, S=1.0, C=1.0, eta=1.0, psi=1.0, k0=1.0):
    """Calculate catalytic activity for DCR"""
    R = 8.314
    try:
        return k0 * math.exp(-Ea / (R * T)) * S * C * eta * psi
    except OverflowError:
        return 0.0

def dcr_routing(G, source, target, T=300):
    """Dynamic Chemical Reaction based routing algorithm"""
    try:
        paths = list(nx.all_simple_paths(G, source=source, target=target, cutoff=6))
    except nx.NetworkXNoPath:
        return []
    
    if not paths:
        return []
    
    best_A, best_path = -1, []
    for path in paths:
        phi = normalize_factors(path, G)
        Ea = activation_energy(phi, weight_vector)
        A = catalytic_activity(Ea, T=T)
        if A > best_A:
            best_A = A
            best_path = path
    
    return best_path

# ----------------------------
# Evaluation Function
# ----------------------------
def evaluate_routing(G, algo_fn, trials=50):
    """Evaluate routing algorithm performance"""
    total_hops = 0
    success = 0
    all_paths = []
    
    for _ in range(trials):
        s, t = random.sample(list(G.nodes()), 2)
        path = algo_fn(G, s, t)
        if path and len(path) > 1:
            all_paths.append((s, t, path))
            total_hops += len(path) - 1
            success += 1
    
    avg_hops = total_hops / success if success > 0 else float('inf')
    return {
        "success_rate": success / trials,
        "avg_hops": avg_hops,
        "paths": all_paths
    }

# ----------------------------
# Utility Functions
# ----------------------------
def compute_path_cost(G, path):
    """Calculate the total cost of a path"""
    if not path or len(path) < 2:
        return float('inf')
    return sum(G[path[i]][path[i + 1]]['weight'] for i in range(len(path) - 1))

def get_best_path(paths, G):
    """Get the best path based on minimum hops"""
    if not paths:
        return []
    return min(paths, key=lambda p: len(p[2]))[2]

# ----------------------------
# Generate Network and Run Tests
# ----------------------------
num_nodes = 50
G = generate_topology(n=num_nodes, p=p, seed=42)
pos = nx.spring_layout(G, seed=42)  # Consistent layout for all visualizations

print(f"Generated network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

# Run evaluation tests
results = {
    "OSPF": evaluate_routing(G, ospf_routing),
    "ACO": evaluate_routing(G, aco_routing),
    "Q-Routing": evaluate_routing(G, q_routing),
    "DCR": evaluate_routing(G, dcr_routing),
}

# ----------------------------
# Display Results Table
# ----------------------------
df = pd.DataFrame({
    "Algorithm": list(results.keys()),
    "Success Rate (%)": [round(results[k]['success_rate']*100, 1) for k in results],
    "Average Hops": [round(results[k]['avg_hops'], 2) for k in results]
})
print("\n" + "="*50)
print("ROUTING ALGORITHM PERFORMANCE COMPARISON")
print("="*50)
print(df.to_string(index=False))
print("="*50)

# ----------------------------
# Enhanced Overlay Visualization
# ----------------------------
# Pick consistent source and target for comparison
source = 0
target = num_nodes - 1

# Ensure source and target are valid
if source >= G.number_of_nodes():
    source = 0
if target >= G.number_of_nodes():
    target = G.number_of_nodes() - 1

print(f"\nGenerating overlay visualization from node {source} to node {target}")

# Run all routing algorithms on the same source-target pair
all_paths = {
    'OSPF': ospf_routing(G, source, target),
    'ACO': aco_routing(G, source, target),
    'Q-Routing': q_routing(G, source, target),
    'DCR': dcr_routing(G, source, target)
}

# Create overlay visualization
plt.figure(figsize=(16, 12))

# Draw base graph
nx.draw_networkx_nodes(G, pos, node_size=50, node_color='lightgrey', alpha=0.7)
nx.draw_networkx_edges(G, pos, edge_color='lightgrey', alpha=0.3, width=0.5)

# Overlay each path with different color and style
line_styles = ['-', '--', '-.', ':']
legend_elements = []

for i, (algo, path) in enumerate(all_paths.items()):
    if path and len(path) > 1:
        edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=edges, 
                             width=3, 
                             edge_color=algo_colors[algo], 
                             alpha=0.8)
        
        # Add path cost to legend
        cost = compute_path_cost(G, path)
        legend_elements.append(f"{algo} (Cost: {cost:.1f}, Hops: {len(path)-1})")
    else:
        legend_elements.append(f"{algo} (No path found)")

# Highlight source and target
nx.draw_networkx_nodes(G, pos, nodelist=[source], node_color='yellow', 
                      node_size=300, edgecolors='black', linewidths=2)
nx.draw_networkx_nodes(G, pos, nodelist=[target], node_color='red', 
                      node_size=300, edgecolors='black', linewidths=2)

# Add labels for source and target
nx.draw_networkx_labels(G, pos, {source: f'S({source})', target: f'T({target})'}, 
                       font_size=12, font_weight='bold')

# Create custom legend
from matplotlib.lines import Line2D
legend_lines = [Line2D([0], [0], color=algo_colors[algo.split()[0]], lw=3) 
                for algo in legend_elements]
plt.legend(legend_lines, legend_elements, 
          title="Routing Algorithms", 
          loc="upper center", 
          bbox_to_anchor=(0.5, -0.05),
          ncol=2, 
          fontsize=12)

plt.title(f"Overlay of Routing Paths on Shared Topology ({load_label})\n"
          f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges", 
          fontsize=14, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.show()

# ----------------------------
# Individual Path Visualizations
# ----------------------------
def draw_individual_path(G, path, algo_name, color, pos):
    """Draw individual algorithm path"""
    plt.figure(figsize=(10, 8))
    
    # Draw base graph
    nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=300, alpha=0.7)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.3)
    nx.draw_networkx_labels(G, pos, font_color='black', font_size=8)
    
    # Highlight path
    if path and len(path) > 1:
        edge_path = list(zip(path, path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=edge_path, edge_color=color, width=4, alpha=0.8)
        nx.draw_networkx_nodes(G, pos, nodelist=path, node_color=color, node_size=400, alpha=0.9)
        
        # Highlight source and target
        nx.draw_networkx_nodes(G, pos, nodelist=[path[0]], node_color='yellow', 
                              node_size=500, edgecolors='black', linewidths=2)
        nx.draw_networkx_nodes(G, pos, nodelist=[path[-1]], node_color='red', 
                              node_size=500, edgecolors='black', linewidths=2)
        
        cost = compute_path_cost(G, path)
        title = f"{algo_name} - Path Cost: {cost:.2f}, Hops: {len(path)-1}"
    else:
        title = f"{algo_name} - No path found"
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Draw individual paths
print("\nGenerating individual path visualizations...")
for algo_name, path in all_paths.items():
    draw_individual_path(G, path, algo_name, algo_colors[algo_name], pos)

# ----------------------------
# Performance Analysis Charts
# ----------------------------
# Success Rate Chart
plt.figure(figsize=(10, 6))
bars = plt.bar(df["Algorithm"], df["Success Rate (%)"], 
               color=[algo_colors[algo] for algo in df["Algorithm"]], 
               alpha=0.7, edgecolor='black')
plt.title(f"Success Rate Comparison ({load_label})", fontsize=14, fontweight='bold')
plt.ylabel("Success Rate (%)", fontsize=12)
plt.ylim(0, 110)
plt.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# Average Hops Chart
plt.figure(figsize=(10, 6))
bars = plt.bar(df["Algorithm"], df["Average Hops"], 
               color=[algo_colors[algo] for algo in df["Algorithm"]], 
               alpha=0.7, edgecolor='black')
plt.title(f"Average Hops Comparison ({load_label})", fontsize=14, fontweight='bold')
plt.ylabel("Average Hops", fontsize=12)
plt.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{height:.2f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# ----------------------------
# Network Topology Overview
# ----------------------------
plt.figure(figsize=(12, 10))
nx.draw_networkx(G, pos, with_labels=True, node_color='lightblue', 
                 edge_color='gray', node_size=200, font_size=8, alpha=0.8)

# Show edge weights (sample for readability)
if G.number_of_edges() < 100:  # Only show weights for smaller networks
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

plt.title(f"Network Topology ({load_label})\n"
          f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
          f"Average degree: {2*G.number_of_edges()/G.number_of_nodes():.2f}", 
          fontsize=14, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.show()

print(f"\nAnalysis complete! Network statistics:")
print(f"- Nodes: {G.number_of_nodes()}")
print(f"- Edges: {G.number_of_edges()}")
print(f"- Average degree: {2*G.number_of_edges()/G.number_of_nodes():.2f}")
print(f"- Network density: {nx.density(G):.4f}")
print(f"- Is connected: {nx.is_connected(G)}")