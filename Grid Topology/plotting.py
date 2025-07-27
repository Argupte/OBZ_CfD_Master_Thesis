import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.spatial import Delaunay
from matplotlib import colormaps

# Define File Directory
base_dir = "E:/TU Delft/Thesis/Code/grid_topology/Reference_Case"

# Load CSV files with semicolon delimiter
df_nodes = pd.read_csv(os.path.join(base_dir, "node_info.csv"), delimiter=";")
df_ac = pd.read_csv(os.path.join(base_dir, "line_info.csv"), delimiter=";")  # AC lines
df_dc = pd.read_csv(os.path.join(base_dir, "DC_info.csv"), delimiter=";")  # DC lines
df_gen = pd.read_csv(os.path.join(base_dir, "gen_info.csv"), delimiter=";")  # Generation data

# Strip spaces from column names
for df in [df_nodes, df_ac, df_dc, df_gen]:
    df.columns = df.columns.str.strip()

# Convert Bus IDs and Zones to appropriate types
df_nodes["BusID"] = df_nodes["BusID"].astype(int)
df_nodes["Zone"] = df_nodes["Zone"].astype(str)  # Zones as strings for labels
df_ac[["FromBus", "ToBus"]] = df_ac[["FromBus", "ToBus"]].astype(int)
df_dc[["FromBus", "ToBus"]] = df_dc[["FromBus", "ToBus"]].astype(int)
df_gen["OnBus"] = df_gen["OnBus"].astype(int)  # Bus where generation occurs

# Identify only zones that have generation
buses_with_generation = df_gen["OnBus"].unique()
zones_with_generation = df_nodes[df_nodes["BusID"].isin(buses_with_generation)]["Zone"].unique()
zones_with_generation = sorted(zones_with_generation)  # Sorted for consistency

# Fix Matplotlib Deprecation Warning
zone_colors = colormaps.get_cmap("tab10")

# **Map Energy Types to Colors**
energy_colors = {
    "PV": "yellow",
    "Wind": "blue",
    "Wind Offshore": "deepskyblue",
    "Nuclear": "limegreen",
    "Gas/CCGT": "darkgray",
    "Hard Coal": "black",
    "Lignite": "darkred",
    "Mixed": "purple"
}

# **Step 1: Identify the centroid of Zone 4**
zone_4_nodes = df_nodes[df_nodes["Zone"] == "4"]
if not zone_4_nodes.empty:
    zone_4_centre_x = zone_4_nodes["Longitude"].mean()
    zone_4_centre_y = zone_4_nodes["Latitude"].mean()
else:
    zone_4_centre_x, zone_4_centre_y = 0, 0  # Default if Zone 4 not found

# **Step 2: Center Zone 4 by shifting all coordinates**
pos_shifted = {}
for _, row in df_nodes.iterrows():
    bus_id = row["BusID"]
    new_x = row["Longitude"] - zone_4_centre_x
    new_y = row["Latitude"] - zone_4_centre_y
    pos_shifted[bus_id] = (new_x, new_y)

# **Function to fill the area enclosed by AC lines for each zone using Delaunay triangulation**
def fill_zone_ac_boundary(zone, df_nodes, df_ac, pos_shifted, color, alpha=0.3):
    """Fill the area enclosed by AC lines within the same zone using Delaunay triangulation."""
    zone_nodes = df_nodes[df_nodes["Zone"] == zone]["BusID"].values
    zone_edges = df_ac[(df_ac["FromBus"].isin(zone_nodes)) & (df_ac["ToBus"].isin(zone_nodes))]

    if len(zone_edges) < 3:
        return  # Not enough edges to form an enclosed region

    # Collect all unique nodes participating in AC edges for this zone
    boundary_nodes = np.unique(zone_edges[["FromBus", "ToBus"]].values.flatten())

    # Get positions of boundary nodes
    boundary_positions = np.array([pos_shifted[n] for n in boundary_nodes if n in pos_shifted])

    if len(boundary_positions) < 3:
        return  # Not enough points for a polygon

    try:
        # Compute Delaunay triangulation to shade the area correctly
        triangulation = Delaunay(boundary_positions)
        for simplex in triangulation.simplices:
            pts = boundary_positions[simplex]
            plt.fill(*zip(*pts), color=color, alpha=alpha, edgecolor="none")
    except:
        pass  # Skip zones where triangulation fails

# **Plot the Grid**
plt.figure(figsize=(14, 10))

# **Fill each zone based on AC line boundaries**
for i, zone in enumerate(zones_with_generation):
    fill_zone_ac_boundary(zone, df_nodes, df_ac, pos_shifted, color=zone_colors(i / len(zones_with_generation)))

# **Predefine generator sizes for the plot**
fixed_size_map_plot = {
    (0, 500): 40,
    (500, 1000): 60,
    (1000, 1500): 80,
    (1500, 2000): 100,
    (2000, float("inf")): 120
}

# **Determine Node Colors & Sizes for Generator Nodes**
node_colors = {}
node_sizes = {}
nodes_with_generators = set()

for _, row in df_nodes.iterrows():
    bus_id = row["BusID"]
    gen_data = df_gen[df_gen["OnBus"] == bus_id]

    if gen_data.empty:
        continue  # Skip for now

    total_capacity = gen_data["Pmax"].sum()
    gen_types = gen_data["Type"].unique()

    # Assign color
    if len(gen_types) == 1:
        node_colors[bus_id] = energy_colors.get(gen_types[0], "gray")
    else:
        node_colors[bus_id] = energy_colors["Mixed"]

    # Assign predefined size based on capacity
    for (low, high), size in fixed_size_map_plot.items():
        if low <= total_capacity < high:
            node_sizes[bus_id] = size
            break

    nodes_with_generators.add(bus_id)

# Create Graph
G = nx.Graph()
for bus_id in df_nodes["BusID"]:
    G.add_node(bus_id)

# Add AC and DC Lines
ac_edges = [(row["FromBus"], row["ToBus"]) for _, row in df_ac.iterrows()]
dc_edges = [(row["FromBus"], row["ToBus"]) for _, row in df_dc.iterrows()]

# Draw AC and DC edges
nx.draw_networkx_edges(G, pos_shifted, edgelist=ac_edges, edge_color="black", width=0.8, label="AC Line")
nx.draw_networkx_edges(G, pos_shifted, edgelist=dc_edges, edge_color="yellow", width=1.5, label="DC Line")

# **Draw Generator Nodes**
nx.draw_networkx_nodes(G, pos_shifted, nodelist=list(nodes_with_generators),
                       node_color=[node_colors[n] for n in nodes_with_generators], 
                       node_size=[node_sizes[n] for n in nodes_with_generators], edgecolors="black")

# **Ensure Compact Legend (2 Columns)**
legend_patches = [
    plt.Line2D([0], [0], marker='o', color='w', label=f"Zone {zone}", 
               markerfacecolor=zone_colors(i / len(zones_with_generation)), markersize=6)
    for i, zone in enumerate(zones_with_generation)
]
legend_patches.extend([
    plt.Line2D([0], [0], marker='o', color='w', label=name, markerfacecolor=color, markersize=6)
    for name, color in energy_colors.items()
])
legend_patches.append(plt.Line2D([0], [0], marker='_', color="black", label="AC Line", linewidth=2))
legend_patches.append(plt.Line2D([0], [0], marker='_', color="yellow", label="DC Line", linewidth=2))

plt.legend(handles=legend_patches, title="Grid Components", loc="upper left", fontsize=8, frameon=True, ncol=2)

plt.title("Power Grid Topology (Labeled Generators & Zones, DC in Yellow, AC in Black)")
plt.axis("off")
plt.show()
