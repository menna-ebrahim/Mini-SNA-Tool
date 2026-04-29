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
    G = build_network(df_edges, df_nodes, choice)
    
    # Summary Metrics
    st.write(f"### Current Statistics:")
    st.info(f"The network has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
# ----- 4. Network Metrics & Analysis Section -----
    st.markdown("---")
    st.header("🔎 Network Metrics & Analysis")

    # --- PageRank Section ---
    trigger_pr = st.button("Run PageRank Analysis")
    if trigger_pr:
        with st.spinner("Processing PageRank computation..."):
            # حساب القيم
            pr_results = nx.pagerank(G)
            nx.set_node_attributes(G, values=pr_results, name="PageRank")

            # تجهيز الجدول
            df_pr = pd.DataFrame(
                [(node, score) for node, score in pr_results.items()],
                columns=["Node", "Score"]
            )

            # إضافة الأسماء
            node_labels = nx.get_node_attributes(G, "Label")
            if len(node_labels) > 0:
                df_pr["Node Name"] = df_pr["Node"].map(node_labels)
                df_pr = df_pr[["Node", "Node Name", "Score"]]

            df_pr = df_pr.sort_values(by="Score", ascending=False).reset_index(drop=True)

            # عرض النتائج
            st.subheader("🏆 Top Influencers (PageRank)")
            st.dataframe(df_pr.head(10))
            st.bar_chart(data=df_pr.head(10), x="Node Name", y="Score")
            st.success("PageRank computation completed!")

    # --- Betweenness Section ---
    trigger_bc = st.button("Run Betweenness Centrality Analysis")
    st.info("💡 High Betweenness Score means this person acts as a 'Bridge' between different groups.")
    
    if trigger_bc:
        with st.spinner("Finding the bridges in the network..."):
            # حساب القيم
            bc_results = nx.betweenness_centrality(G)
            nx.set_node_attributes(G, values=bc_results, name="Betweenness")

            # تجهيز الجدول
            df_bc = pd.DataFrame(
                [(node, score) for node, score in bc_results.items()],
                columns=["Node", "Betweenness Score"]
            )

            # إضافة الأسماء
            node_labels = nx.get_node_attributes(G, "Label")
            if len(node_labels) > 0:
                df_bc["Node Name"] = df_bc["Node"].map(node_labels)
                df_bc = df_bc[["Node", "Node Name", "Betweenness Score"]]

            df_bc = df_bc.sort_values(by="Betweenness Score", ascending=False).reset_index(drop=True)

            # عرض النتائج
            st.subheader("🌉 Top Information Brokers (Betweenness)")
            st.dataframe(df_bc.head(10))
            st.bar_chart(data=df_bc.head(10), x="Node Name", y="Betweenness Score")
            st.success("Betweenness Centrality computation completed!")

    # --- 5. Quick Search (Bonus Feature) ---
    st.markdown("---")
    search_node = st.text_input("🔍 Search for a node score by Name (Works after running analysis):")
    if search_node:
        # البحث هنا يتم في بيانات الـ Nodes الأصلية المرفوعة
        search_res = df_nodes[df_nodes['Label'].str.contains(search_node, case=False, na=False)]
        if not search_res.empty:
            st.write("Node Details Found:", search_res)
        else:
            st.warning("Node not found in the uploaded data.")