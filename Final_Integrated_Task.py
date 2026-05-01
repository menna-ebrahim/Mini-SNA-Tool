import streamlit as st
import networkx as nx
import pandas as pd
import community as community_louvain
from sklearn.metrics import silhouette_score, normalized_mutual_info_score
import numpy as np
from pyvis.network import Network
import streamlit.components.v1 as components

# --- 1. Basic Configuration ---
st.set_page_config(layout="wide", page_title="Mini Gephi Tool - Project Dashboard") 

# --- [الزيادة: Metrics Master Functions] ---
def calculate_global_metrics(G):
    metrics = {}
    metrics['Avg. Degree'] = sum(dict(G.degree()).values()) / G.number_of_nodes()
    # حساب طول المسار (لأكبر مكون متصل لتجنب الأخطاء)
    if nx.is_connected(G if not G.is_directed() else G.to_undirected()):
        metrics['Avg. Path Length'] = nx.average_shortest_path_length(G)
    else:
        largest_cc = max(nx.connected_components(G.to_undirected()), key=len)
        metrics['Avg. Path Length (LCC)'] = nx.average_shortest_path_length(G.subgraph(largest_cc))
    metrics['Density'] = nx.density(G)
    return metrics

def build_network(e_data, n_data, graph_type):
    """Function to build the network graph from dataframes"""
    if graph_type == "Directed":
        my_graph = nx.DiGraph()
    else:
        my_graph = nx.Graph()
    for _, edge in e_data.iterrows():
        my_graph.add_edge(edge['source'], edge['target'])
    for _, node in n_data.iterrows():
        n_id = node['Id']
        if n_id in my_graph:
            attrs = node.to_dict()
            my_graph.nodes[n_id].update(attrs)
    return my_graph

def check_detection():
    """Check if community detection has been run before evaluating"""
    if 'partition' not in st.session_state:
        st.error("⚠️ Please run 'Detect Communities' first to generate data!")
        return False
    return True

st.title("🌐 Mini Gephi Tool - Project Dashboard")
st.divider()

# --- 2. Screen Split (Left Menu & Main Workspace) ---
col_left_menu, col_main = st.columns([1, 3])

# --- 3. Left Menu (Control Panel) ---
with col_left_menu:
    st.header("🎛️ Control Panel")
    
    with st.expander("📂 1. Data Import", expanded=True):
        n_upload = st.file_uploader("Nodes (CSV)", type=["csv"])
        e_upload = st.file_uploader("Edges (CSV)", type=["csv"])
        gt_upload = st.file_uploader("Ground Truth (Optional CSV)", type=["csv"])
        graph_type = st.radio("Type", ["Undirected", "Directed"])

    with st.expander("🎯 2. Filtering Options", expanded=True):
        deg_filter = st.slider("Minimum Node Degree", 0, 50, 0)

    with st.expander("🚀 3. Link Analysis", expanded=True):
        btn_pagerank = st.button("Run PageRank")
        btn_betweenness = st.button("Run Betweenness")

    # --- [الزيادة: Metrics Master Control] ---
    with st.expander("📈 4. Metrics Master", expanded=True):
        btn_global = st.button("Calculate Global Metrics")
        btn_centralities = st.button("Run Degree & Closeness Centrality")

    with st.expander("🏘️ 5. Community Detection", expanded=True):
        algo_choice = st.selectbox("Select Algorithm", ["Louvain", "Girvan-Newman"])
        btn_detect = st.button("Detect Communities")
        
        st.divider()
        st.write("**Evaluation Metrics**")
        btn_modularity = st.button("1. Modularity")
        btn_silhouette = st.button("2. Silhouette Score")
        btn_nmi = st.button("3. NMI Score")
        
        st.divider()
        btn_compare = st.button("📊 Compare Algorithms")

# --- 4. Main Workspace (Results Area) ---
with col_main:
    st.subheader("🖥️ Workspace Results")
    
    if n_upload and e_upload:
        # قراءة البيانات
        df_nodes = pd.read_csv(n_upload)
        df_edges = pd.read_csv(e_upload)
        raw_Graph = build_network(df_edges, df_nodes, graph_type)
        
        degrees = dict(raw_Graph.degree())
        to_keep = [n for n, d in degrees.items() if d >= deg_filter]
        final_Graph = raw_Graph.subgraph(to_keep)
        
        st.success(f"✔️ Network built successfully! Showing {final_Graph.number_of_nodes()} nodes after filtering.")

        tab_preview, tab_viz, tab_analysis, tab_metrics, tab_comm = st.tabs([
            "📊 Data Preview", "🕸️ Interactive Viz", "🚀 PageRank & Betweenness Results", "📈 Metrics Master Results", "🏘️ Community Detection Results"
        ])

        # --- Tab 1: Data Preview ---
        with tab_preview:
            col1, col2 = st.columns(2)
            col1.metric("Nodes Count", final_Graph.number_of_nodes())
            col2.metric("Edges Count", final_Graph.number_of_edges())
            st.write("### Nodes Data Table")
            st.dataframe(df_nodes.head(10), use_container_width=True)
            st.write("### Edges Data Table")
            st.dataframe(df_edges.head(10), use_container_width=True)

        # --- Tab 2: Interactive Visualization (Pyvis) ---
        with tab_viz:
            st.write("Drag nodes to interact. Zoom in/out using the mouse wheel.")
            net = Network(height="500px", width="100%", bgcolor="#222222", font_color="white")
            
            node_labels = nx.get_node_attributes(final_Graph, "Label")
            
            for n in final_Graph.nodes():
                lbl = node_labels.get(n, str(n))
                node_size = degrees.get(n, 1) * 5
                net.add_node(str(n), label=lbl, size=node_size, title=f"Node ID: {n} | Connections: {degrees.get(n)}", color="#00ffcc")
            
            for u, v in final_Graph.edges():
                net.add_edge(str(u), str(v))

            net.save_graph("network_map.html")
            with open("network_map.html", 'r', encoding='utf-8') as f:
                components.html(f.read(), height=550)

        # --- Tab 3: Link Analysis Processing ---
        with tab_analysis:
            node_labels = nx.get_node_attributes(final_Graph, "Label")
            
            if btn_pagerank:
                with st.spinner("Calculating PageRank..."):
                    pr = nx.pagerank(final_Graph)
                    df_pr = pd.DataFrame(list(pr.items()), columns=["Node", "Score"])
                    if node_labels:
                        df_pr["Node Name"] = df_pr["Node"].map(node_labels)
                    df_pr = df_pr.sort_values(by="Score", ascending=False).reset_index(drop=True)
                    st.write("### 🏆 PageRank Results (Top 10)")
                    st.dataframe(df_pr.head(10), use_container_width=True)
                    st.bar_chart(data=df_pr.head(10), x="Node Name" if node_labels else "Node", y="Score")
                    
            if btn_betweenness:
                with st.spinner("Calculating Betweenness Centrality..."):
                    bc = nx.betweenness_centrality(final_Graph)
                    df_bc = pd.DataFrame(list(bc.items()), columns=["Node", "Score"])
                    if node_labels:
                        df_bc["Node Name"] = df_bc["Node"].map(node_labels)
                    df_bc = df_bc.sort_values(by="Score", ascending=False).reset_index(drop=True)
                    st.write("### 🌉 Betweenness Centrality Results (Top 10)")
                    st.dataframe(df_bc.head(10), use_container_width=True)
                    st.bar_chart(data=df_bc.head(10), x="Node Name" if node_labels else "Node", y="Score")
            
            if not btn_pagerank and not btn_betweenness:
                st.info("Click a button from the 'Link Analysis' panel to see results here.")

        # --- [الزيادة: Tab 4: Metrics Master Results] ---
        with tab_metrics:
            if btn_global:
                with st.spinner("Calculating Global Metrics..."):
                    m = calculate_global_metrics(final_Graph)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Avg. Degree", f"{m['Avg. Degree']:.2f}")
                    c2.metric("Density", f"{m['Density']:.4f}")
                    path_label = 'Avg. Path Length' if 'Avg. Path Length' in m else 'Avg. Path Length (LCC)'
                    c3.metric(path_label, f"{m[path_label]:.2f}")

            if btn_centralities:
                with st.spinner("Calculating Centralities..."):
                    d_cent = nx.degree_centrality(final_Graph)
                    c_cent = nx.closeness_centrality(final_Graph)
                    df_metrics = pd.DataFrame({
                        "Node": list(d_cent.keys()),
                        "Degree Centrality": list(d_cent.values()),
                        "Closeness Centrality": list(c_cent.values())
                    })
                    if node_labels:
                        df_metrics["Node Name"] = df_metrics["Node"].map(node_labels)
                    st.write("### 📊 Centrality Suite (Top 10 by Degree)")
                    st.dataframe(df_metrics.sort_values(by="Degree Centrality", ascending=False).head(10), use_container_width=True)
            
            if not btn_global and not btn_centralities:
                st.info("Click a button from the 'Metrics Master' panel to see results here.")

        # --- Tab 5: Community Detection Processing ---
        with tab_comm:
            if btn_detect:
                with st.spinner(f"Running {algo_choice}..."):
                    working_G = final_Graph.to_undirected() if final_Graph.is_directed() else final_Graph
                    if algo_choice == "Louvain":
                        partition = community_louvain.best_partition(working_G)
                    else:
                        from networkx.algorithms.community import girvan_newman
                        partition = {node: i for i, c in enumerate(next(girvan_newman(working_G))) for node in c}
                    
                    st.session_state['partition'] = partition
                    st.session_state['last_algo'] = algo_choice
                    st.session_state['show_detection'] = True

            if 'show_detection' in st.session_state:
                st.write(f"### 🏘️ Algorithm Used: {st.session_state['last_algo']}")
                df_res = pd.DataFrame(list(st.session_state['partition'].items()), columns=["Node", "Community ID"])
                st.dataframe(df_res.head(10), use_container_width=True)

            if btn_modularity and check_detection():
                working_G = final_Graph.to_undirected() if final_Graph.is_directed() else final_Graph
                score = community_louvain.modularity(st.session_state['partition'], working_G)
                st.metric("Modularity Score", f"{score:.4f}")

            if btn_silhouette and check_detection():
                adj_matrix = nx.to_numpy_array(final_Graph)
                labels = list(st.session_state['partition'].values())
                if len(set(labels)) > 1:
                    score = silhouette_score(adj_matrix, labels)
                    st.metric("Silhouette Score", f"{score:.4f}")
                else: 
                    st.warning("Needs > 1 community to calculate Silhouette.")

            if btn_nmi and check_detection():
                if gt_upload:
                    df_gt = pd.read_csv(gt_upload)
                    gt_dict = dict(zip(df_gt['Id'], df_gt['Community']))
                    common = [n for n in final_Graph.nodes() if n in gt_dict]
                    if common:
                        score = normalized_mutual_info_score([gt_dict[n] for n in common], [st.session_state['partition'][n] for n in common])
                        st.metric("NMI Score", f"{score:.4f}")
                    else:
                        st.error("No matching Node IDs between Ground Truth and Data.")
                else: 
                    st.warning("Please upload Ground Truth file to calculate NMI.")

            if btn_compare:
                with st.spinner("Calculating comprehensive comparison..."):
                    working_G = final_Graph.to_undirected() if final_Graph.is_directed() else final_Graph
                    adj_matrix = nx.to_numpy_array(final_Graph)
                    
                    # Louvain
                    p_louvain = community_louvain.best_partition(working_G)
                    m_louvain = community_louvain.modularity(p_louvain, working_G)
                    s_louvain = silhouette_score(adj_matrix, list(p_louvain.values())) if len(set(p_louvain.values())) > 1 else 0
                    
                    # Girvan-Newman
                    from networkx.algorithms.community import girvan_newman
                    p_gn = {node: i for i, c in enumerate(next(girvan_newman(working_G))) for node in c}
                    m_gn = community_louvain.modularity(p_gn, working_G)
                    s_gn = silhouette_score(adj_matrix, list(p_gn.values())) if len(set(p_gn.values())) > 1 else 0

                    nmi_louvain, nmi_gn = "N/A", "N/A"
                    if gt_upload:
                        df_gt = pd.read_csv(gt_upload)
                        gt_dict = dict(zip(df_gt['Id'], df_gt['Community']))
                        common = [n for n in final_Graph.nodes() if n in gt_dict]
                        if common:
                            nmi_louvain = f"{normalized_mutual_info_score([gt_dict[n] for n in common], [p_louvain[n] for n in common]):.4f}"
                            nmi_gn = f"{normalized_mutual_info_score([gt_dict[n] for n in common], [p_gn[n] for n in common]):.4f}"

                    st.write("### ⚖️ Comprehensive Algorithm Comparison")
                    comp_df = pd.DataFrame({
                        "Metric": ["Communities Found", "Modularity (Internal)", "Silhouette (Internal)", "NMI (External)"],
                        "Louvain": [len(set(p_louvain.values())), f"{m_louvain:.4f}", f"{s_louvain:.4f}", nmi_louvain],
                        "Girvan-Newman": [len(set(p_gn.values())), f"{m_gn:.4f}", f"{s_gn:.4f}", nmi_gn]
                    })
                    st.table(comp_df)

        # --- 5. Quick Search Feature (Task 6 Bonus) ---
        st.divider()
        st.header("🔍 Quick Node Lookup")
        search_node = st.text_input("Search for a node by Label (e.g., Name):")
        if search_node:
            search_res = df_nodes[df_nodes['Label'].str.contains(search_node, case=False, na=False)]
            if not search_res.empty:
                st.write("✅ Node Details Found:", search_res)
            else:
                st.warning("❌ Node not found in the uploaded data.")

    else:
        st.warning("👈 Please upload Nodes and Edges to begin.")