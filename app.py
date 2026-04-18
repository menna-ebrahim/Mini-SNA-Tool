import streamlit as st
import pandas as pd
import networkx as nx

# --- Configuration ---
st.set_page_config(layout="wide", page_title="SNA Project") 

def build_network(e_data, n_data, graph_type):
    """
    Function to create the graph object based on user input
    """
    if graph_type == "Directed":
        my_graph = nx.DiGraph()
    else:
        my_graph = nx.Graph()
        
    # Loading edges
    for _, edge in e_data.iterrows():
        my_graph.add_edge(edge['source'], edge['target'])
        
    # Mapping node attributes
    for _, node in n_data.iterrows():
        n_id = node['Id']
        if n_id in my_graph:
            # Updating node with all extra info from CSV
            attrs = node.to_dict()
            my_graph.nodes[n_id].update(attrs)
            
    return my_graph

# --- Sidebar UI ---
st.sidebar.subheader("Configuration Panel")
n_upload = st.sidebar.file_uploader("Upload Nodes Data", type=["csv"])
e_upload = st.sidebar.file_uploader("Upload Edges Data", type=["csv"])

# --- Main App Logic ---
st.title("Social Network Analysis Tool")
st.info("Upload your CSV files from the sidebar to begin analyzing the graph.")

if n_upload and e_upload:
    # Read CSVs
    df_nodes = pd.read_csv(n_upload)
    df_edges = pd.read_csv(e_upload)
    
    st.success("Data imported successfully!")
    
    # Display Tabs 
    tab1, tab2 = st.tabs(["Nodes Table", "Edges Table"])
    with tab1:
        st.dataframe(df_nodes.head(5)) 
    with tab2:
        st.dataframe(df_edges.head(5))

    # Graph Type Selection
    choice = st.sidebar.selectbox("Graph Type", ["Undirected", "Directed"]) 
    
    # Generate the Graph from CSV Files
    final_Graph = build_network(df_edges, df_nodes, choice)
    
    # Summary Metrics
    st.write(f"### Current Statistics:")
    st.info(f"The network has {final_Graph.number_of_nodes()} nodes and {final_Graph.number_of_edges()} edges.")
    