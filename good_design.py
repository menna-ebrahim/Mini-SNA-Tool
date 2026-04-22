import streamlit as st
import networkx as nx
import pandas as pd
import community as community_louvain
from sklearn.metrics import silhouette_score, normalized_mutual_info_score
import numpy as np

# --- 1. الإعدادات الأساسية ---
st.set_page_config(layout="wide", page_title="Mini Gephi Tool - Community Edition") 

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

def check_detection():
    if 'partition' not in st.session_state:
        st.error("⚠️ Please run 'Detect Communities' first to generate data!")
        return False
    return True

st.title("🌐 Mini Gephi Tool - Project Dashboard")
st.divider()

# --- 2. تقسيم الشاشة ---
col_left_menu, col_main = st.columns([1, 3])

# --- 3. القائمة اليسرى ---
with col_left_menu:
    st.header("🎛️ Control Panel")
    
    with st.expander("📂 1. Data Import", expanded=True):
        n_upload = st.file_uploader("Nodes (CSV)", type=["csv"])
        e_upload = st.file_uploader("Edges (CSV)", type=["csv"])
        gt_upload = st.file_uploader("Ground Truth (Optional CSV)", type=["csv"])
        graph_type = st.radio("Type", ["Undirected", "Directed"])

    with st.expander("🏘️ 2. Community Detection", expanded=True):
        algo_choice = st.selectbox("Select Algorithm", ["Louvain", "Girvan-Newman"])
        btn_detect = st.button("Detect Communities")
        
        st.divider()
        st.write("**Evaluation Metrics**")
        btn_modularity = st.button("1. Modularity")
        btn_silhouette = st.button("2. Silhouette Score")
        btn_nmi = st.button("3. NMI Score")
        
        st.divider()
        btn_compare = st.button("📊 Compare Algorithms (All Metrics)")

# --- 4. منطقة النتائج ---
with col_main:
    st.subheader("🖥️ Workspace Results")
    
    if n_upload and e_upload:
        df_nodes = pd.read_csv(n_upload)
        df_edges = pd.read_csv(e_upload)
        final_Graph = build_network(df_edges, df_nodes, graph_type)
        st.success(f"✔️ Network built: {final_Graph.number_of_nodes()} nodes")

        # --- تشغيل الاكتشاف ---
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

        # --- عرض النتائج الثابتة ---
        if 'show_detection' in st.session_state:
            st.write(f"### 🏘️ Last Algorithm: {st.session_state['last_algo']}")
            df_res = pd.DataFrame(list(st.session_state['partition'].items()), columns=["Node", "Community ID"])
            st.dataframe(df_res.head(10), use_container_width=True)
            st.write("---")

        # --- أزرار التقييم الفردية ---
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
            else: st.warning("Needs > 1 community.")

        if btn_nmi and check_detection():
            if gt_upload:
                df_gt = pd.read_csv(gt_upload)
                gt_dict = dict(zip(df_gt['Id'], df_gt['Community']))
                common = [n for n in final_Graph.nodes() if n in gt_dict]
                score = normalized_mutual_info_score([gt_dict[n] for n in common], [st.session_state['partition'][n] for n in common])
                st.metric("NMI Score", f"{score:.4f}")
            else: st.warning("Please upload Ground Truth file.")

        # --- المقارنة الشاملة (المطلوبة) ---
        if btn_compare:
            with st.spinner("Calculating comparison for all metrics..."):
                working_G = final_Graph.to_undirected() if final_Graph.is_directed() else final_Graph
                adj_matrix = nx.to_numpy_array(final_Graph)
                
                # --- Louvain Calculations ---
                p_louvain = community_louvain.best_partition(working_G)
                m_louvain = community_louvain.modularity(p_louvain, working_G)
                s_louvain = silhouette_score(adj_matrix, list(p_louvain.values())) if len(set(p_louvain.values())) > 1 else 0
                
                # --- Girvan-Newman Calculations ---
                from networkx.algorithms.community import girvan_newman
                p_gn = {node: i for i, c in enumerate(next(girvan_newman(working_G))) for node in c}
                m_gn = community_louvain.modularity(p_gn, working_G)
                s_gn = silhouette_score(adj_matrix, list(p_gn.values())) if len(set(p_gn.values())) > 1 else 0

                # --- NMI Calculations (If file exists) ---
                nmi_louvain, nmi_gn = "N/A", "N/A"
                if gt_upload:
                    df_gt = pd.read_csv(gt_upload)
                    gt_dict = dict(zip(df_gt['Id'], df_gt['Community']))
                    common = [n for n in final_Graph.nodes() if n in gt_dict]
                    actual = [gt_dict[n] for n in common]
                    nmi_louvain = f"{normalized_mutual_info_score(actual, [p_louvain[n] for n in common]):.4f}"
                    nmi_gn = f"{normalized_mutual_info_score(actual, [p_gn[n] for n in common]):.4f}"

                # --- Display Table ---
                st.write("### ⚖️ Comprehensive Algorithm Comparison")
                comp_df = pd.DataFrame({
                    "Metric": ["Communities Found", "Modularity (Internal)", "Silhouette (Internal)", "NMI (External/Ground Truth)"],
                    "Louvain": [len(set(p_louvain.values())), f"{m_louvain:.4f}", f"{s_louvain:.4f}", nmi_louvain],
                    "Girvan-Newman": [len(set(p_gn.values())), f"{m_gn:.4f}", f"{s_gn:.4f}", nmi_gn]
                })
                st.table(comp_df)
                if not gt_upload:
                    st.caption("ℹ️ Upload a Ground Truth file to see NMI scores in the comparison.")

    else:
        st.warning("👈 Please upload Nodes and Edges to begin.")