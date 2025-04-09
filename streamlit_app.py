import streamlit as st
import networkx as nx
import random
import matplotlib.pyplot as plt
import tempfile
from pyvis.network import Network
import os

st.set_page_config(page_title="Idiomapp", layout="wide")

def create_graph(graph_type, num_nodes, randomize_edges=False):
    """Create different types of graphs based on user selection"""
    
    if graph_type == "Path":
        G = nx.path_graph(num_nodes)
    elif graph_type == "Cycle":
        G = nx.cycle_graph(num_nodes)
    elif graph_type == "Star":
        G = nx.star_graph(num_nodes - 1)
    elif graph_type == "Complete":
        G = nx.complete_graph(num_nodes)
    elif graph_type == "Barabasi-Albert":
        # For Barabasi-Albert, m must be at least 1 and less than n
        m = min(3, num_nodes - 1)
        if m > 0:
            G = nx.barabasi_albert_graph(num_nodes, m)
        else:
            G = nx.path_graph(num_nodes)  # Fallback
    else:
        G = nx.path_graph(num_nodes)  # Default
    
    # Add some random edges if requested
    if randomize_edges and num_nodes > 2:
        num_random_edges = random.randint(1, 3)
        for _ in range(num_random_edges):
            a, b = random.sample(list(G.nodes()), 2)
            if not G.has_edge(a, b):
                G.add_edge(a, b)
    
    return G

def visualize_graph_pyvis(G, central_node=None):
    """Create an interactive visualization of the graph using Pyvis"""
    
    # Create a network
    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
    
    # Set options
    net.barnes_hut()
    net.set_options("""
    {
      "nodes": {
        "borderWidth": 2,
        "borderWidthSelected": 4,
        "color": {
          "border": "#FFFFFF",
          "background": "#69b3a2"
        },
        "font": {
          "size": 15,
          "face": "Tahoma"
        },
        "shadow": true
      },
      "edges": {
        "color": {
          "color": "#FFFFFF",
          "highlight": "#FF8C00",
          "hover": "#FF8C00"
        },
        "smooth": false,
        "shadow": true,
        "width": 2
      },
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -8000,
          "springConstant": 0.04,
          "springLength": 95
        },
        "stabilization": {
          "iterations": 1000
        }
      },
      "interaction": {
        "dragNodes": true,
        "hideEdgesOnDrag": false,
        "hideNodesOnDrag": false
      }
    }
    """)
    
    # Add nodes and edges to the network
    for node in G.nodes():
        # Special color for central node
        if central_node is not None and node == central_node:
            net.add_node(node, label=f"Node {node}", title=f"Node {node}", color="#ff3e3e", size=25)
        else:
            net.add_node(node, label=f"Node {node}", title=f"Node {node}", size=20)
    
    for edge in G.edges():
        net.add_edge(edge[0], edge[1])
    
    # Create a temporary HTML file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmpfile:
        path = tmpfile.name
        net.save_graph(path)
    
    # Display the graph
    with open(path, 'r', encoding='utf-8') as f:
        html_string = f.read()
    
    # Clean up the temp file
    os.unlink(path)
    
    # Display the network
    st.components.v1.html(html_string, height=610)

def node_analytics(G):
    """Display analytics about the graph"""
    
    st.subheader("Graph Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Number of Nodes", len(G.nodes()))
        st.metric("Number of Edges", len(G.edges()))
        
        if len(G.nodes()) > 0:
            # Calculate degree centrality
            degree_centrality = nx.degree_centrality(G)
            most_central_node = max(degree_centrality, key=degree_centrality.get)
            st.metric("Most Central Node", f"Node {most_central_node}")
    
    with col2:
        # Check if the graph is connected
        is_connected = nx.is_connected(G) if len(G.nodes()) > 0 else False
        st.metric("Connected", "Yes" if is_connected else "No")
        
        # Average clustering
        try:
            avg_clustering = nx.average_clustering(G)
            st.metric("Average Clustering", f"{avg_clustering:.4f}")
        except:
            st.metric("Average Clustering", "N/A")
    
    return most_central_node if len(G.nodes()) > 0 else None

def main():
    st.title("Idiomapp")
    
    # Add a sidebar for controls
    with st.sidebar:
        st.header("Graph Controls")
        
        graph_type = st.selectbox(
            "Graph Type",
            ["Path", "Cycle", "Star", "Complete", "Barabasi-Albert"]
        )
        
        num_nodes = st.slider("Number of Nodes", 3, 20, 8)
        
        randomize_edges = st.checkbox("Add Random Connections", value=False)
        
        highlight_central = st.checkbox("Highlight Central Node", value=True)
        
        if st.button("Generate New Graph"):
            st.session_state["graph"] = create_graph(graph_type, num_nodes, randomize_edges)
    
    # Initialize graph if it doesn't exist in session state
    if "graph" not in st.session_state:
        st.session_state["graph"] = create_graph("Path", 8)
    
    # Two columns layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Display graph analytics
        most_central_node = node_analytics(st.session_state["graph"])
        
        # Only highlight if requested
        central_node = most_central_node if highlight_central else None
        
        # Display the graph
        visualize_graph_pyvis(st.session_state["graph"], central_node=central_node)
    
    with col2:
        st.markdown("""
        ### About this Visualization
        
        This interactive graph visualization allows you to explore different graph structures.
        
        #### Interaction:
        - **Drag nodes** to reposition them
        - **Zoom** with mouse wheel
        - **Pan** by dragging the background
        
        #### Graph Types:
        - **Path**: Linear sequence of nodes
        - **Cycle**: Circular sequence of nodes
        - **Star**: One central node connected to all others
        - **Complete**: Every node connected to every other node
        - **Barabasi-Albert**: Scale-free network model
        
        Adjust the controls in the sidebar to create different types of graphs.
        """)

if __name__ == "__main__":
    main() 