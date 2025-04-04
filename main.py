import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import random
import time
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="CNCMP Network Simulator",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stApp {
        background-color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #f5f5f5;
        color: #000000;
    }
    .st-bq {
        border-left: 3px solid #4e79a7;
    }
    .metric-card {
        background: #f5f5f5;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 15px;
        border: 1px solid #ddd;
    }
    .metric-title {
        color: #555;
        font-size: 14px;
        font-weight: 600;
    }
    .metric-value {
        color: #000000;
        font-size: 24px;
        font-weight: 700;
    }
    .tab-container {
        background: #f5f5f5;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #ddd;
    }
    .stDataFrame {
        background-color: #f5f5f5 !important;
    }
    .st-eb {
        background-color: #f5f5f5;
    }
    .st-cb {
        background-color: #f5f5f5;
    }
    .css-1aumxhk {
        background-color: #f5f5f5;
    }
</style>
""", unsafe_allow_html=True)
# -----------------------------
# Sidebar Inputs (Enhanced)
# -----------------------------
with st.sidebar:
    st.image("https://via.placeholder.com/150x50?text=CNCMP+Sim", width=150)
    st.title("Network Parameters")
    
    with st.expander("📊 Network Configuration", expanded=True):
        NUM_NODES = st.slider("Number of Nodes", 2, 50, 10, help="Total nodes in the network")
        NUM_LINKS = st.slider("Number of Links", 1, 100, 15, help="Total connections between nodes")
        LINK_DENSITY = st.slider("Link Density", 0.1, 1.0, 0.3, help="Probability of connection between nodes")
        
    with st.expander("📨 Traffic Configuration"):
        NUM_REQUESTS = st.slider("Number of Requests", 1, 50, 5)
        MAX_CAPACITY = st.slider("Max Link Capacity", 5, 100, 20, step=5)
        MAX_LATENCY = st.slider("Max Allowed Latency", 5, 50, 15)
        TRAFFIC_VARIATION = st.slider("Traffic Variation", 0.1, 1.0, 0.5, help="How much demand varies between requests")
    
    with st.expander("⚙️ Simulation Options"):
        ROUTING_STRATEGY = st.selectbox(
            "Routing Strategy",
            ["Cost-Aware", "Latency-Optimal", "Load-Balanced", "Random"],
            help="Algorithm for path selection"
        )
        VISUALIZATION_STYLE = st.selectbox(
            "Visualization Style",
            ["Spring Layout", "Circular Layout", "Kamada-Kawai", "Shell Layout"],
            index=0
        )
    
    run_sim = st.button("🚀 Run Simulation", use_container_width=True)
    st.markdown("---")
    st.caption("CNCMP Network Simulator v2.0")

# -----------------------------
# Main Page Layout
# -----------------------------
st.title("🌐 CNCMP Network Simulation")
st.caption("Simulate network traffic routing with different strategies")

if not run_sim:
    # Show welcome content before simulation runs
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        ### Welcome to CNCMP Network Simulator
        
        This tool helps you simulate network traffic routing in a configurable data center environment.
        
        **Features:**
        - Generate random network topologies
        - Create traffic requests with varying demands
        - Visualize network utilization and latency
        - Multiple routing strategies
        
        Configure your simulation in the sidebar and click **Run Simulation** to start.
        """)
        
        with st.expander("📚 Simulation Methodology"):
            st.markdown("""
            The simulation follows these steps:
            1. **Network Generation**: Creates a random graph with specified nodes and links
            2. **Link Properties**: Assigns capacity and latency values to each link
            3. **Request Generation**: Creates traffic demands between random node pairs
            4. **Routing**: Finds paths based on selected strategy
            5. **Visualization**: Shows network state with color-coded metrics
            """)
            
    with col2:
    # Use HTML with iframe for better animation support
        st.components.v1.html("""
    <div style="background-color: #1a1a1a; border-radius: 10px; padding: 20px; width: 100%; height: 300px;">
        <svg width="100%" height="100%" viewBox="0 0 400 300" xmlns="http://www.w3.org/2000/svg">
            <!-- Network nodes -->
            <circle cx="100" cy="100" r="15" fill="#e74c3c" stroke="white" stroke-width="2"/>
            <circle cx="200" cy="80" r="15" fill="#3498db" stroke="white" stroke-width="2"/>
            <circle cx="300" cy="100" r="15" fill="#2ecc71" stroke="white" stroke-width="2"/>
            <circle cx="150" cy="180" r="15" fill="#3498db" stroke="white" stroke-width="2"/>
            <circle cx="250" cy="180" r="15" fill="#2ecc71" stroke="white" stroke-width="2"/>
            
            <!-- Network connections -->
            <line x1="100" y1="100" x2="200" y2="80" stroke="#00ff00" stroke-width="3">
                <animate attributeName="stroke-opacity" values="0.3;1;0.3" dur="2s" repeatCount="indefinite"/>
            </line>
            <line x1="200" y1="80" x2="300" y2="100" stroke="#ffff00" stroke-width="3">
                <animate attributeName="stroke-opacity" values="0.3;1;0.3" dur="2s" repeatCount="indefinite" begin="0.5s"/>
            </line>
            <line x1="100" y1="100" x2="150" y2="180" stroke="#ff0000" stroke-width="3">
                <animate attributeName="stroke-opacity" values="0.3;1;0.3" dur="2s" repeatCount="indefinite" begin="1s"/>
            </line>
            <line x1="150" y1="180" x2="250" y2="180" stroke="#00ff00" stroke-width="3">
                <animate attributeName="stroke-opacity" values="0.3;1;0.3" dur="2s" repeatCount="indefinite" begin="1.5s"/>
            </line>
            <line x1="250" y1="180" x2="300" y2="100" stroke="#ffff00" stroke-width="3">
                <animate attributeName="stroke-opacity" values="0.3;1;0.3" dur="2s" repeatCount="indefinite" begin="2s"/>
            </line>
            
            <!-- Node labels -->
            <text x="100" y="100" font-size="12" fill="white" text-anchor="middle" dy="30">Core</text>
            <text x="200" y="80" font-size="12" fill="white" text-anchor="middle" dy="30">Agg</text>
            <text x="300" y="100" font-size="12" fill="white" text-anchor="middle" dy="30">Edge</text>
            <text x="150" y="180" font-size="12" fill="white" text-anchor="middle" dy="30">Agg</text>
            <text x="250" y="180" font-size="12" fill="white" text-anchor="middle" dy="30">Edge</text>
        </svg>
    </div>
    """, height=320)

    # Loading animation using pure CSS
        st.markdown("""
    <style>
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .loader-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin-top: 20px;
        }
        .loader {
            border: 5px solid #f3f3f3;
            border-top: 5px solid #3498db;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 2s linear infinite;
        }
        .loader-text {
            color: white;
            margin-top: 10px;
        }
    </style>
    <div class="loader-container">
        <div class="loader"></div>
        <p class="loader-text">Preparing network simulation...</p>
    </div>
         """, unsafe_allow_html=True)
    
        st.caption("Interactive visualization will appear here after simulation")
    
    st.stop()

# -----------------------------
# Simulation Backend Logic
# -----------------------------

# Initialize metrics dictionary
metrics = {
    "total_requests": NUM_REQUESTS,
    "successful_routes": 0,
    "blocked_requests": 0,
    "avg_latency": 0,
    "max_utilization": 0,
    "total_capacity": 0,
    "used_capacity": 0
}

# Progress bar
progress_bar = st.progress(0)
status_text = st.empty()

# Step 1: Build Random Network
status_text.text("Step 1/4: Building network topology...")
progress_bar.progress(10)

# Use different graph generation methods based on density
if LINK_DENSITY > 0.7:
    G = nx.dense_gnm_random_graph(NUM_NODES, NUM_LINKS)
else:
    G = nx.gnp_random_graph(NUM_NODES, LINK_DENSITY)

# Assign node types (core, aggregation, edge)
for node in G.nodes():
    if random.random() < 0.2:
        G.nodes[node]['type'] = 'core'
    elif random.random() < 0.5:
        G.nodes[node]['type'] = 'aggregation'
    else:
        G.nodes[node]['type'] = 'edge'

# Assign link properties
for u, v in G.edges():
    # Base capacity influenced by node types
    base_cap = MAX_CAPACITY
    if G.nodes[u]['type'] == 'core' or G.nodes[v]['type'] == 'core':
        base_cap *= 1.5
    G.edges[u, v]['capacity'] = max(5, int(random.normalvariate(base_cap, base_cap*0.2)))
    
    # Base latency influenced by node types
    base_lat = 2 if (G.nodes[u]['type'] == 'core' and G.nodes[v]['type'] == 'core') else 5
    G.edges[u, v]['latency'] = random.randint(base_lat, base_lat + 8)
    G.edges[u, v]['used_capacity'] = 0
    G.edges[u, v]['failures'] = 0

metrics['total_capacity'] = sum(G.edges[u, v]['capacity'] for u, v in G.edges())

# Select layout based on user choice
status_text.text("Step 2/4: Calculating network layout...")
progress_bar.progress(30)

layout_options = {
    "Spring Layout": nx.spring_layout,
    "Circular Layout": nx.circular_layout,
    "Kamada-Kawai": nx.kamada_kawai_layout,
    "Shell Layout": nx.shell_layout
}
pos = layout_options[VISUALIZATION_STYLE](G, seed=42)

# Step 2: Generate Random Requests
status_text.text("Step 3/4: Generating traffic requests...")
progress_bar.progress(50)

requests = []
for _ in range(NUM_REQUESTS):
    src = random.randint(0, NUM_NODES - 1)
    dst = random.randint(0, NUM_NODES - 1)
    while dst == src:
        dst = random.randint(0, NUM_NODES - 1)
    
    # Demand based on traffic variation parameter
    base_demand = MAX_CAPACITY * 0.3
    demand = max(1, int(random.normalvariate(base_demand, base_demand * TRAFFIC_VARIATION)))
    
    # Max latency based on network diameter
    diameter = nx.diameter(G) if nx.is_connected(G) else 10
    max_allowed_latency = random.randint(diameter * 2, diameter * 2 + MAX_LATENCY)
    
    requests.append((src, dst, demand, max_allowed_latency))

# Display requests in a nice table
with st.expander("📋 Generated Traffic Requests", expanded=True):
    request_df = pd.DataFrame(requests, columns=["Source", "Destination", "Demand", "Max Latency"])
    st.dataframe(request_df.style.background_gradient(cmap="Blues"), use_container_width=True)

# Step 3: Routing Logic
status_text.text("Step 4/4: Routing traffic...")
progress_bar.progress(70)

result_text = ""
routing_results = []

def calculate_path_metrics(G, path, demand):
    path_latency, path_capacity, path_cost = 0, float('inf'), 0
    for u, v in zip(path[:-1], path[1:]):
        link = G.edges[u, v]
        utilization = link['used_capacity'] / link['capacity'] if link['capacity'] > 0 else 1
        queue_delay = utilization * 5  # Simple queueing model
        total_link_latency = link['latency'] + queue_delay
        path_latency += total_link_latency
        path_capacity = min(path_capacity, link['capacity'] - link['used_capacity'])
        path_cost += total_link_latency + (10 * utilization)  # Cost function
        
    return path_latency, path_capacity, path_cost

for i, (src, dst, demand, max_latency) in enumerate(requests):
    try:
        all_paths = list(nx.all_simple_paths(G, src, dst, cutoff=6))
        valid_paths = []
        
        for path in all_paths:
            path_latency, path_capacity, path_cost = calculate_path_metrics(G, path, demand)
            
            if path_latency <= max_latency and path_capacity >= demand:
                valid_paths.append((path, path_latency, path_capacity, path_cost))
        
        if valid_paths:
            # Select path based on routing strategy
            if ROUTING_STRATEGY == "Cost-Aware":
                best_path = sorted(valid_paths, key=lambda x: x[3])[0]  # Lowest cost
            elif ROUTING_STRATEGY == "Latency-Optimal":
                best_path = sorted(valid_paths, key=lambda x: x[1])[0]  # Lowest latency
            elif ROUTING_STRATEGY == "Load-Balanced":
                best_path = sorted(valid_paths, key=lambda x: -x[2])[0]  # Max capacity
            else:  # Random
                best_path = random.choice(valid_paths)
                
            path = best_path[0]
            # Update link utilization
            for u, v in zip(path[:-1], path[1:]):
                G.edges[u, v]['used_capacity'] += demand
                G.edges[u, v]['failures'] = 0
            
            result_text += f"✅ Request {i+1}: {src}→{dst} via {path} | demand={demand} | latency={round(best_path[1],2)} | cost={round(best_path[3],2)}\n"
            routing_results.append({
                "Request": f"{src}→{dst}",
                "Status": "Success",
                "Path": path,
                "Demand": demand,
                "Latency": round(best_path[1], 2),
                "Cost": round(best_path[3], 2),
                "Hops": len(path)-1
            })
            metrics['successful_routes'] += 1
            metrics['avg_latency'] += best_path[1]
        else:
            result_text += f"⚠️ Request {i+1}: {src}→{dst} BLOCKED (No feasible path for demand={demand})\n"
            routing_results.append({
                "Request": f"{src}→{dst}",
                "Status": "Blocked",
                "Path": [],
                "Demand": demand,
                "Latency": 0,
                "Cost": 0,
                "Hops": 0
            })
            metrics['blocked_requests'] += 1
            
            # Update failure counters on potential paths
            for path in all_paths:
                for u, v in zip(path[:-1], path[1:]):
                    G.edges[u, v]['failures'] += 1

    except nx.NetworkXNoPath:
        result_text += f"❌ Request {i+1}: {src}→{dst} BLOCKED (No path exists)\n"
        routing_results.append({
            "Request": f"{src}→{dst}",
            "Status": "No Path",
            "Path": [],
            "Demand": demand,
            "Latency": 0,
            "Cost": 0,
            "Hops": 0
        })
        metrics['blocked_requests'] += 1
    
    progress_bar.progress(70 + int(30 * (i+1)/NUM_REQUESTS))

# Calculate final metrics
if metrics['successful_routes'] > 0:
    metrics['avg_latency'] /= metrics['successful_routes']
metrics['used_capacity'] = sum(G.edges[u, v]['used_capacity'] for u, v in G.edges())
metrics['max_utilization'] = max(
    (G.edges[u, v]['used_capacity'] / G.edges[u, v]['capacity'] 
     for u, v in G.edges() if G.edges[u, v]['capacity'] > 0),
    default=0
)

progress_bar.progress(100)
status_text.text("Simulation complete!")

# -----------------------------
# Results Display
# -----------------------------

# Metrics cards
st.subheader("📊 Simulation Metrics")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Success Rate</div>
        <div class="metric-value">{metrics['successful_routes']}/{metrics['total_requests']}</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Avg Latency</div>
        <div class="metric-value">{round(metrics['avg_latency'], 2)} ms</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Max Utilization</div>
        <div class="metric-value">{round(metrics['max_utilization']*100, 1)}%</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Capacity Used</div>
        <div class="metric-value">{round(metrics['used_capacity']/metrics['total_capacity']*100, 1)}%</div>
    </div>
    """, unsafe_allow_html=True)

# Routing results in tabs
tab1, tab2, tab3 = st.tabs(["📝 Routing Results", "📈 Network Analysis", "🌐 Visualization"])

with tab1:
    st.subheader("Routing Results")
    st.code(result_text, language='text')
    
    # Detailed results table
    # Detailed results table
    results_df = pd.DataFrame(routing_results)
    st.dataframe(
        results_df.style.map(
            lambda x: (
                'background-color: #e6f3e6; color: #2e7d32'  # Dark green text on light green background
                if x == 'Success' 
                else 'background-color: #ffebee; color: #c62828'  # Dark red text on light red background
            ),
            subset=['Status']
        ).set_properties(**{
        'font-weight': 'bold',
        'text-align': 'center'
        }),
        use_container_width=True
    )
with tab2:
    st.subheader("Network Analysis")
    
    # Link utilization distribution
    utilizations = [
        G.edges[u, v]['used_capacity'] / G.edges[u, v]['capacity'] 
        for u, v in G.edges() if G.edges[u, v]['capacity'] > 0
    ]
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Link Utilization Distribution**")
        fig, ax = plt.subplots()
        ax.hist(utilizations, bins=10, color='#3498db', edgecolor='#2980b9')
        ax.set_xlabel('Utilization (%)')
        ax.set_ylabel('Number of Links')
        ax.set_title('Link Utilization Histogram')
        st.pyplot(fig)
    
    with col2:
        st.markdown("**Latency vs. Utilization**")
        latencies = [G.edges[u, v]['latency'] for u, v in G.edges()]
        util_pct = [u*100 for u in utilizations]
        
        fig, ax = plt.subplots()
        sc = ax.scatter(util_pct, latencies, c=latencies, cmap='viridis')
        ax.set_xlabel('Utilization (%)')
        ax.set_ylabel('Base Latency (ms)')
        ax.set_title('Latency vs Utilization')
        plt.colorbar(sc, label='Latency (ms)')
        st.pyplot(fig)
    
    # Node centrality measures
    st.markdown("**Node Centrality Measures**")
    degree_centrality = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)
    
    centrality_df = pd.DataFrame({
        'Node': list(degree_centrality.keys()),
        'Degree Centrality': list(degree_centrality.values()),
        'Betweenness': list(betweenness.values()),
        'Closeness': list(closeness.values()),
        'Type': [G.nodes[n]['type'] for n in G.nodes()]
    })
    
    st.dataframe(
        centrality_df.style.background_gradient(cmap='Blues', subset=['Degree Centrality', 'Betweenness', 'Closeness']),
        use_container_width=True
    )

with tab3:
    st.subheader("Network Visualization")
    
    # Create two columns for layout
    col_viz, col_controls = st.columns([3, 1])
    
    with col_controls:
        st.markdown("### Visualization Controls")
        # Animation is now always enabled
        animation_speed = st.slider("Animation Speed", 1, 10, 5, 
                                  help="Adjust the speed of the traffic flow animation")
        
        if metrics['successful_routes'] > 0:
            st.markdown("---")
            st.markdown("### Path Explorer")
            selected_request = st.selectbox(
                "Select request:",
                [f"Request {i+1}: {r['Request']}" for i, r in enumerate(routing_results) if r['Status'] == 'Success']
            )

    with col_viz:
        # Create custom colormaps
        cmap = LinearSegmentedColormap.from_list('traffic', ['#00ff00', '#ffff00', '#ff0000'])
        node_cmap = {'core': '#e74c3c', 'aggregation': '#3498db', 'edge': '#2ecc71'}
        
        # Calculate metrics
        betweenness = nx.betweenness_centrality(G)
        node_sizes = [2000 * (betweenness[n] + 0.1) for n in G.nodes()]
        node_colors = [node_cmap[G.nodes[n]['type']] for n in G.nodes()]
        
        # Create figure with dark background
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='black')
        ax.set_facecolor('black')
        
        # Draw edges with base colors
        nx.draw_networkx_edges(
            G, pos,
            width=[1 + 4 * (G.edges[u, v]['used_capacity'] / G.edges[u, v]['capacity']) 
                  for u, v in G.edges()],
            edge_color=[cmap(G.edges[u, v]['used_capacity'] / G.edges[u, v]['capacity']) 
                       for u, v in G.edges()],
            alpha=0.7,
            ax=ax
        )
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color=node_colors,
            edgecolors='white',
            linewidths=1,
            alpha=0.9,
            ax=ax
        )
        
        # Draw labels with white text
        nx.draw_networkx_labels(
            G, pos,
            font_size=10,
            font_weight='bold',
            font_color='white',
            ax=ax
        )
        
        # Create legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label='Core', 
                      markerfacecolor=node_cmap['core'], markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Aggregation', 
                      markerfacecolor=node_cmap['aggregation'], markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Edge', 
                      markerfacecolor=node_cmap['edge'], markersize=10),
            plt.Line2D([0], [0], color='#00ff00', lw=2, label='Low Util'),
            plt.Line2D([0], [0], color='#ffff00', lw=2, label='Medium Util'),
            plt.Line2D([0], [0], color='#ff0000', lw=2, label='High Util')
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', facecolor='black', labelcolor='white')
        ax.set_title(f"CNCMP Network State ({ROUTING_STRATEGY} Routing)", color='white')
        
        # Show the static visualization
        viz_placeholder = st.empty()
        viz_placeholder.pyplot(fig)
        
        # Create animated edges (always enabled now)
        progress_text = st.empty()
        progress_text.text("Initializing network traffic animation...")
        
        animated_edges = []
        for u, v in G.edges():
            edge, = ax.plot([], [], color='cyan', linewidth=2, alpha=0.7)
            animated_edges.append((u, v, edge))
        
        # Run the animation indefinitely
        while True:
            for i in range(100):
                # Update animation frames
                for u, v, edge in animated_edges:
                    x = np.linspace(pos[u][0], pos[v][0], 10)
                    y = np.linspace(pos[u][1], pos[v][1], 10)
                    edge.set_data(x[:i%10+1], y[:i%10+1])
                
                # Redraw the figure
                viz_placeholder.pyplot(fig)
                time.sleep(0.1 / animation_speed)
            
            # Show selected path if available (without breaking the animation)
            if metrics['successful_routes'] > 0 and 'selected_request' in locals():
                req_idx = int(selected_request.split()[1].replace(":", "")) - 1
                path = routing_results[req_idx]['Path']
                
                # Create path visualization
                fig_path, ax_path = plt.subplots(figsize=(10, 8), facecolor='black')
                ax_path.set_facecolor('black')
                
                # Draw base graph
                nx.draw_networkx_nodes(
                    G, pos, 
                    node_size=300, 
                    node_color=['gray' if n not in path else node_cmap[G.nodes[n]['type']] for n in G.nodes()],
                    edgecolors='white',
                    linewidths=1,
                    ax=ax_path
                )
                nx.draw_networkx_edges(
                    G, pos,
                    width=1,
                    edge_color='gray',
                    alpha=0.3,
                    ax=ax_path
                )
                
                # Highlight path
                path_edges = list(zip(path[:-1], path[1:]))
                nx.draw_networkx_edges(
                    G, pos,
                    edgelist=path_edges,
                    width=3,
                    edge_color='#e74c3c',
                    alpha=0.8,
                    ax=ax_path
                )
                
                # Draw labels
                nx.draw_networkx_labels(
                    G, pos,
                    font_size=10,
                    font_color='white',
                    ax=ax_path
                )
                
                ax_path.set_title(f"Path for {selected_request}", color='white')
                st.pyplot(fig_path)
