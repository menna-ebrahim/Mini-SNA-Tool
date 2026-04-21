import streamlit as st
import networkx as nx
import pandas as pd

# --- الإعدادات الأساسية ---
st.set_page_config(layout="wide", page_title="Mini Gephi Tool") 

def build_network(e_data, n_data, graph_type):
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

st.title("🌐 Mini Gephi Tool")
st.divider()

# ==========================================
# 1. تقسيم الشاشة: الشمال للقائمة واليمين للنتائج
# ==========================================
col_left_menu, col_main = st.columns([1, 3])

# ==========================================
# 2. العمود الشمال (لوحة التحكم - Control Panel)
# ==========================================
with col_left_menu:
    st.header("🎛️ Control Panel")
    
    # --- القسم الأول: رفع الملفات ---
    with st.expander("📂 1. Data Import", expanded=True):
        n_upload = st.file_uploader("Nodes (CSV)", type=["csv"])
        e_upload = st.file_uploader("Edges (CSV)", type=["csv"])
        graph_type = st.radio("Type", ["Undirected", "Directed"])

    # --- القسم الثاني: مقاييس الرسم البياني ---
    with st.expander("📊 2. Graph Metrics"):
        btn_degree_dist = st.button("Degree Distribution")
        btn_avg_path = st.button("Average Path Length")

    # --- القسم الثالث: تحليل الروابط (Link Analysis) ---
    with st.expander("🔗 3. Link Analysis"):
        btn_pagerank = st.button("Run PageRank Analysis")
        btn_betweenness = st.button("Run Betweenness Analysis")

    # --- القسم الرابع: اكتشاف المجتمعات والتقييم (التفاصيل كاملة هنا) ---
    with st.expander("🏘️ 4. Community Detection"):
        st.write("**1. Partitioning (Clustering)**")
        algo_choice = st.selectbox("Algorithm", ["Louvain", "Girvan-Newman"])
        btn_detect = st.button("Detect Communities")
        
        st.divider()
        st.write("**2. Clustering Evaluation**")
        btn_modularity = st.button("1. Modularity (Internal)")
        btn_silhouette = st.button("2. Silhouette Score (Internal)")
        btn_nmi = st.button("3. NMI Score (External)")

    # --- القسم الخامس: خيارات الفلترة (التفاصيل كاملة هنا) ---
    with st.expander("🔍 5. Filtering Options"):
        st.write("**Filter by Centrality Ranges:**")
        # السلايدرز الثلاثة المطلوبة
        slider_degree = st.slider("Degree Centrality Range", 0.0, 1.0, (0.0, 1.0))
        slider_betweenness = st.slider("Betweenness Range", 0.0, 1.0, (0.0, 1.0))
        slider_pagerank = st.slider("PageRank Range", 0.0, 1.0, (0.0, 1.0))
        
        st.divider()
        st.write("**Filter by Membership:**")
        selected_communities = st.multiselect("Select Communities to display", ["Community 0", "Community 1", "Community 2"])
        
        btn_apply_filters = st.button("Apply Filters", type="primary")

# ==========================================
# 3. العمود اليمين (مساحة العمل والنتائج)
# ==========================================
with col_main:
    st.subheader("🖥️ Workspace")
    
    if n_upload and e_upload:
        # بناء الشبكة
        df_nodes = pd.read_csv(n_upload)
        df_edges = pd.read_csv(e_upload)
        final_Graph = build_network(df_edges, df_nodes, graph_type)
        
        st.success(f"✔️ Network built with {final_Graph.number_of_nodes()} nodes and {final_Graph.number_of_edges()} edges.")
        
        # --- منطق تشغيل PageRank ---
        if btn_pagerank:
                with st.spinner("Processing PageRank..."):
                    pr_results = nx.pagerank(final_Graph)
                    nx.set_node_attributes(final_Graph, values=pr_results, name="PageRank")
                    
                    df_pr = pd.DataFrame([(node, score) for node, score in pr_results.items()], columns=["Node", "Score"])
                    node_labels = nx.get_node_attributes(final_Graph, "Label")
                    if node_labels:
                        df_pr["Node Name"] = df_pr["Node"].map(node_labels)
                        df_pr = df_pr[["Node", "Node Name", "Score"]]
                    
                    df_pr = df_pr.sort_values(by="Score", ascending=False).reset_index(drop=True)
                    
                    st.write("###  Top Ranked Nodes (PageRank)")
                    st.dataframe(df_pr.head(10), use_container_width=True)

            # 3. لو المستخدم داس على زرار Betweenness اللي في الشمال
        if btn_betweenness:
            with st.spinner("Finding bridges..."):
                bc_results = nx.betweenness_centrality(final_Graph)
                
                df_bc = pd.DataFrame([(node, score) for node, score in bc_results.items()], columns=["Node", "Score"])
                node_labels = nx.get_node_attributes(final_Graph, "Label")
                if node_labels:
                    df_bc["Node Name"] = df_bc["Node"].map(node_labels)
                    df_bc = df_bc[["Node", "Node Name", "Score"]]
                
                df_bc = df_bc.sort_values(by="Score", ascending=False).reset_index(drop=True)
                
                st.write("###  Top Information Brokers (Betweenness)")
                st.dataframe(df_bc.head(10), use_container_width=True)

    else:
        st.warning("👈 Please upload your CSV files from the Control Panel to start.")