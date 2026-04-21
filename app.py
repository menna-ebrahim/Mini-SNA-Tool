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
    # ----- Custom Section -----
st.markdown("___")
st.header("🔎 Network Metrics & Analysis")

# زر لتنفيذ العملية عند الطلب
trigger_pr = st.button("Run PageRank Analysis")

if trigger_pr:
    with st.spinner("Processing PageRank computation..."):

        # حساب القيم
        pr_results = nx.pagerank(G)

        # تخزين النتائج داخل الجراف
        nx.set_node_attributes(G, values=pr_results, name="PageRank")

        # تجهيز البيانات للعرض
        df_pr = pd.DataFrame(
            [(node, score) for node, score in pr_results.items()],
            columns=["Node", "Score"]
        )

        # إضافة أسماء (لو موجودة)
        node_labels = nx.get_node_attributes(G, "Label")
        if len(node_labels) > 0:
            df_pr["Node Name"] = df_pr["Node"].map(node_labels)
            df_pr = df_pr[["Node", "Node Name", "Score"]]

        # ترتيب النتائج
        df_pr = df_pr.sort_values(by="Score", ascending=False).reset_index(drop=True)

        # عرض أفضل النتائج
        st.subheader("Top Ranked Nodes")
        st.dataframe(df_pr.iloc[:10])

        st.success("PageRank computation completed and graph updated!")

# --- زر لتنفيذ حساب الـ Betweenness Centrality ---
trigger_bc = st.button("Run Betweenness Centrality Analysis")

if trigger_bc:
    with st.spinner("Finding the bridges in the network..."):

        # 1. حساب القيم
        bc_results = nx.betweenness_centrality(G)

        # 2. تخزين النتائج داخل الجراف
        nx.set_node_attributes(G, values=bc_results, name="Betweenness")

        # 3. تجهيز البيانات للعرض
        df_bc = pd.DataFrame(
            [(node, score) for node, score in bc_results.items()],
            columns=["Node", "Betweenness Score"]
        )

        # 4. إضافة أسماء (لو موجودة)
        node_labels = nx.get_node_attributes(G, "Label")
        if len(node_labels) > 0:
            df_bc["Node Name"] = df_bc["Node"].map(node_labels)
            # إعادة ترتيب الأعمدة
            df_bc = df_bc[["Node", "Node Name", "Betweenness Score"]]

        # 5. ترتيب النتائج من الأعلى للأقل
        df_bc = df_bc.sort_values(by="Betweenness Score", ascending=False).reset_index(drop=True)

        # 6. عرض أفضل النتائج
        st.subheader(" Top Information Brokers (Betweenness)")
        st.dataframe(df_bc.iloc[:10])

        st.success(" Betweenness Centrality computation completed!")        