import streamlit as st
import networkx as nx
import pandas as pd
import community as community_louvain
from sklearn.metrics import silhouette_score, normalized_mutual_info_score
import numpy as np
from pyvis.network import Network
import streamlit.components.v1 as components
import random # [تمت الإضافة] لتوليد ألوان عشوائية للمجموعات

st.set_page_config(layout="wide", page_title="Mini Gephi Tool - Project Dashboard") 

def calculate_global_metrics(G):
    metrics = {}
    # 1. Degree Metrics
    degrees = dict(G.degree()).values()
    metrics['Avg. Degree'] = sum(degrees) / G.number_of_nodes()
    
    # 2. Average Clustering Coefficient (مطلوب)
    # يحسب مدى ترابط جيران العقدة ببعضهم البعض
    metrics['Avg. Clustering Coeff.'] = nx.average_clustering(G.to_undirected())
    
    # 3. Path & Diameter (مطلوب)
    if nx.is_connected(G if not G.is_directed() else G.to_undirected()):
        metrics['Avg. Path Length'] = nx.average_shortest_path_length(G)
        metrics['Network Diameter'] = nx.diameter(G)
    else:
        # حساب القطر للمكون الأكبر في حالة الشبكة غير المتصلة
        largest_cc = G.subgraph(max(nx.connected_components(G.to_undirected()), key=len))
        metrics['Avg. Path Length (LCC)'] = nx.average_shortest_path_length(largest_cc)
        metrics['Network Diameter (LCC)'] = nx.diameter(largest_cc)
        
    metrics['Density'] = nx.density(G)
    return metrics


def build_network(e_data, n_data, graph_type):
    if graph_type == "Directed":
        my_graph = nx.DiGraph()
    else:
        my_graph = nx.Graph()
    for _, edge in e_data.iterrows():
        my_graph.add_edge(edge['Source'], edge['Target'])
    for _, node in n_data.iterrows():
        n_id = node['ID']
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

# دالة مساعدة لتوليد ألوان ثابتة بناءً على القيم
def get_hex_color(val, color_map):
    if val not in color_map:
        color_map[val] = f"#{random.randint(0, 0xFFFFFF):06x}"
    return color_map[val]

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

    with st.expander("🚀 3. Link Analysis", expanded=True):
        btn_pagerank = st.button("Run PageRank")
        btn_betweenness = st.button("Run Betweenness")

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
        # 1. قراءة البيانات الأصلية
        df_nodes = pd.read_csv(n_upload)
        df_edges = pd.read_csv(e_upload)
        
        # --- [التأكد من أعمدة الـ Nodes] ---
        existing_node_cols = [c.lower() for c in df_nodes.columns]
        if 'label' not in existing_node_cols:
            df_nodes['Label'] = "" 


        # --- [التأكد من أعمدة الـ Edges (التعديل الجديد بتاعك)] ---
        existing_edge_cols = [c.lower() for c in df_edges.columns]
        
        # إضافة عمود Weight لو مش موجود (بنديله رقم 1 كبداية)
        if 'weight' not in existing_edge_cols:
            df_edges['Weight'] = 1.0
            
        # إضافة عمود Label للعلاقات لو مش موجود (بنسيبه فاضي)
        if 'label' not in existing_edge_cols:
            df_edges['Label'] = ""
        # -----------------------------------------------------

        # 2. حفظ البيانات في الـ session_state عشان الـ Data Laboratory يشتغل صح
        if 'edit_nodes' not in st.session_state:
            st.session_state['edit_nodes'] = df_nodes.copy()
        if 'edit_edges' not in st.session_state:
            st.session_state['edit_edges'] = df_edges.copy()

        # 3. بناء الـ Graph باستخدام الداتا القابلة للتعديل
        raw_Graph = build_network(st.session_state['edit_edges'], st.session_state['edit_nodes'], graph_type)
        
        # --- 4. حساب الـ Metrics عشان نستخدمها في الفلاتر ---
        deg_cent = nx.degree_centrality(raw_Graph)
        bet_cent = nx.betweenness_centrality(raw_Graph)
        pr_cent = nx.pagerank(raw_Graph)
        
        undirected_G = raw_Graph.to_undirected() if raw_Graph.is_directed() else raw_Graph
        communities = community_louvain.best_partition(undirected_G)

        # 5. إضافة القيم دي جوه الـ Graph لكل Node
        nx.set_node_attributes(raw_Graph, deg_cent, 'Degree_Cent')
        nx.set_node_attributes(raw_Graph, bet_cent, 'Betweenness')
        nx.set_node_attributes(raw_Graph, pr_cent, 'PageRank')
        nx.set_node_attributes(raw_Graph, communities, 'Community')



        # --- 2. بناء واجهة الفلاتر والستايل في القائمة الجانبية ---
        with col_left_menu:
            with st.expander("🎯 2. Advanced Filtering Options", expanded=False):
                st.write("**Filter by Centrality Ranges:**")
                deg_range = st.slider("Degree Centrality", 0.0, max(deg_cent.values()), (0.0, max(deg_cent.values())))
                bet_range = st.slider("Betweenness Centrality", 0.0, max(bet_cent.values()), (0.0, max(bet_cent.values())))
                pr_range = st.slider("PageRank Score", 0.0, max(pr_cent.values()), (0.0, max(pr_cent.values())))
                
                st.divider()
                st.write("**Filter by Community:**")
                unique_comms = list(set(communities.values()))
                selected_comms = st.multiselect("Select Communities:", unique_comms, default=unique_comms)

            # --- [تمت الإضافة: 6. Visual Styling] ---
            with st.expander("🎨 6. Visual Styling (Gephi-like)", expanded=True):
                global_node_color = st.color_picker("Pick Default Node Color", "#1f78b4")
    
    # 2. التلوين بناءً على خاصية (Partition/Ranking)
                
                # استخراج أسماء الأعمدة من النودز للتلون بيها (ماعدا الـ ID)
                cat_cols = [c for c in df_nodes.columns if c.lower() not in ['id', 'label']]
                color_options = ["Default", "Community"] + cat_cols
                
                color_choice = st.selectbox("Color Nodes By:", color_options)
                size_choice = st.selectbox("Size Nodes By:", ["Default", "Degree Centrality", "Betweenness", "PageRank"])
                shape_choice = st.selectbox("Node Shape:", ["dot", "square", "triangle", "star", "hexagon"])

        # --- 3. تطبيق الفلاتر واستخراج الـ Graph النهائي ---
        to_keep = []
        for n, data in raw_Graph.nodes(data=True):
            d_val = data.get('Degree_Cent', 0)
            b_val = data.get('Betweenness', 0)
            p_val = data.get('PageRank', 0)
            c_val = data.get('Community', -1)
            
            if (deg_range[0] <= d_val <= deg_range[1] and
                bet_range[0] <= b_val <= bet_range[1] and
                pr_range[0] <= p_val <= pr_range[1] and
                c_val in selected_comms):
                to_keep.append(n)
                
        final_Graph = raw_Graph.subgraph(to_keep)
        degrees = dict(final_Graph.degree())
        
        
        


        st.success(f"✔️ Network built successfully! Showing {final_Graph.number_of_nodes()} nodes after filtering.")

        tab_preview, tab_viz, tab_analysis, tab_metrics, tab_comm = st.tabs([
            "📊 Data Preview", "🕸️ Interactive Viz", "🚀 PageRank & Betweenness Results", "📈 Metrics Master Results", "🏘️ Community Detection Results"
        ])



        # --- Tab 1: Data Laboratory (Gephi-like) ---
        with tab_preview:
            st.header("📝 Data Laboratory")
            st.markdown("Edit your data directly, add new rows, or create new columns.")
            
            # 1. حفظ الجداول في الـ session_state عشان التعديلات ماتضيعش مع الـ Refresh
            if 'edit_nodes' not in st.session_state:
                st.session_state['edit_nodes'] = df_nodes.copy()
            if 'edit_edges' not in st.session_state:
                st.session_state['edit_edges'] = df_edges.copy()

            col1, col2 = st.columns(2)
            col1.metric("Nodes Count", final_Graph.number_of_nodes())
            col2.metric("Edges Count", final_Graph.number_of_edges())
            
            st.divider()

            # ==========================================
            # 🟢 قسم النودز (Nodes Table)
            # ==========================================
            st.subheader("🟢 Nodes Table (Editable)")
            
            # خاصية إضافة عمود جديد (زي Label)
            

            # جدول التعديل التفاعلي (Data Editor)
            st.session_state['edit_nodes'] = st.data_editor(
                st.session_state['edit_nodes'],
                num_rows="dynamic", # الميزة دي بتسمح بإضافة أو مسح صفوف من الجدول مباشرة
                use_container_width=True,
                key="editor_nodes"
            )
            
            st.divider()

            # ==========================================
            # 🔗 قسم العلاقات (Edges Table)
            # ==========================================
            st.subheader("🔗 Edges Table (Editable)")
            
            # جدول التعديل التفاعلي للـ Edges
            st.session_state['edit_edges'] = st.data_editor(
                st.session_state['edit_edges'],
                num_rows="dynamic",
                use_container_width=True,
                key="editor_edges"
            )
            
            # زرار لحفظ التعديلات وتطبيقها على الرسمة
            if st.button("🔄 Apply Changes to Graph", type="primary"):
                # تحديث الداتا الأساسية بالداتا اللي اتعدلت
                df_nodes = st.session_state['edit_nodes']
                df_edges = st.session_state['edit_edges']
                st.success("Changes applied successfully! Go to Interactive Viz to see updates.")
                # ملاحظة: عشان التغيير يظهر في الرسمة فوراً، ممكن تحتاجي تعملي rerun أو تعيدي بناء الـ Graph
                # st.rerun()

        # --- Tab 2: Interactive Visualization (Pyvis) ---
        with tab_viz:
            st.write("Drag nodes to interact. Zoom in/out using the mouse wheel.")
            
            layout_choice = st.selectbox("🕸️ Select Layout Algorithm:", 
                 ["Force-Directed (Fruchterman-Reingold)", 
                  "Circular Layout", 
                  "Random Layout", 
                  "Hierarchical (Tree Layout)", 
                  "Live Physics (Animation - Slow)"])
            
            with st.spinner("Calculating Layout positions..."):
                if layout_choice == "Force-Directed (Fruchterman-Reingold)":
                    pos = nx.spring_layout(final_Graph, seed=42)
                elif layout_choice == "Circular Layout":
                    pos = nx.circular_layout(final_Graph)
                elif layout_choice == "Random Layout":
                    pos = nx.random_layout(final_Graph, seed=42)
                else:
                    pos = None 

            net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
    # تفعيل نظام الطبقات في Pyvis
            if layout_choice == "Hierarchical (Tree Layout)":
                # تفعيل نظام الطبقات في Pyvis
                net.set_options("""
                {
                "layout": {
                    "hierarchical": {
                    "enabled": true,
                    "direction": "UD",
                    "sortMethod": "directed"
                    }
                }
                }
                """)
            node_labels = nx.get_node_attributes(final_Graph, "Label")
            
            # --- [تم التعديل: تطبيق الألوان والأحجام والأشكال ديناميكياً] ---
            dynamic_color_map = {} 

            for n in final_Graph.nodes():
                node_data = final_Graph.nodes[n]
                lbl = node_labels.get(n, str(n))
                
                # 1. تحديد الحجم (Size)
                if size_choice == "Degree Centrality":
                    # نضرب في رقم كبير عشان الحجم يظهر بوضوح
                    node_size = max(node_data.get('Degree_Cent', 0) * 100, 10)
                elif size_choice == "Betweenness":
                    node_size = max(node_data.get('Betweenness', 0) * 300, 10)
                elif size_choice == "PageRank":
                    node_size = max(node_data.get('PageRank', 0) * 500, 10)
                else:
                    node_size = 15 
                
                # 2. تحديد اللون (Color)
                if color_choice == "Community":
                    c_val = node_data.get('Community', 0)
                    node_color = get_hex_color(c_val, dynamic_color_map)
                elif color_choice != "Default":
                    attr_val = node_data.get(color_choice, "Unknown")
                    node_color = get_hex_color(attr_val, dynamic_color_map)
                else:
                   node_color = global_node_color 
                
                # 3. النص التعريفي (Hover text)
                hover_text = f"Node ID: {n}\nLabel: {lbl}\nConnections: {degrees.get(n)}"
                
                # 4. إضافة النقطة للرسمة
                if pos is not None:
                    x_coord = pos[n][0] * 1000
                    y_coord = pos[n][1] * 1000
                    net.add_node(str(n), label=lbl, size=node_size, color=node_color, shape=shape_choice, title=hover_text, x=x_coord, y=y_coord, physics=False)
                else:
                    net.add_node(str(n), label=lbl, size=node_size, color=node_color, shape=shape_choice, title=hover_text)
            
            for u, v, data in final_Graph.edges(data=True):
                edge_width = data.get('Weight', 1.0) 
                # هناخد الـ Label بتاع العلاقة لو موجود
                edge_label = data.get('Label', "")
                
                net.add_edge(str(u), str(v), width=edge_width, label=edge_label)
                

            if pos is not None:
                net.toggle_physics(False)
            else:
                net.toggle_physics(True) 

            net.save_graph("network_map.html")
            with open("network_map.html", 'r', encoding='utf-8') as f:
                components.html(f.read(), height=650)

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

        # --- Tab 4: Metrics Master Results ---
        # داخل tab_metrics:
with tab_metrics:
    if btn_global:
        with st.spinner("Calculating Global Metrics..."):
            m = calculate_global_metrics(final_Graph)
            c1, c2, c3, c4, c5 = st.columns(5) 
                    
            c1.metric("Avg. Degree", f"{m['Avg. Degree']:.2f}")
            c2.metric("Density", f"{m['Density']:.4f}")
            c3.metric("Avg. Clustering", f"{m['Avg. Clustering Coeff.']:.4f}")
            
            # 2. إضافة الـ Average Path Length (سواء للشبكة كاملة أو الـ LCC)
            path_label = 'Avg. Path Length' if 'Avg. Path Length' in m else 'Avg. Path Length (LCC)'
            c4.metric(path_label, f"{m[path_label]:.2f}")
            
            # 3. عرض القطر في العمود الخامس
            diam_label = 'Network Diameter' if 'Network Diameter' in m else 'Network Diameter (LCC)'
            c5.metric(diam_label, m[diam_label])
            
     

            st.divider()
            
            # --- رسم توزيع الدرجات (Degree Distribution) ---
            st.write("### 📊 Degree Distribution")
            degree_sequence = [d for n, d in final_Graph.degree()]
            degree_counts = pd.Series(degree_sequence).value_counts().sort_index()
            df_dist = pd.DataFrame({"Degree": degree_counts.index, "Number of Nodes": degree_counts.values})
            st.bar_chart(df_dist, x="Degree", y="Number of Nodes")

    if btn_centralities:
        with st.spinner("Calculating Centralities..."):
            d_cent = nx.degree_centrality(final_Graph)
            c_cent = nx.closeness_centrality(final_Graph)
            # إضافة الـ Local Clustering لكل Node
            l_clust = nx.clustering(final_Graph.to_undirected())
            
            df_metrics = pd.DataFrame({
                "Node": list(d_cent.keys()),
                "Degree Centrality": list(d_cent.values()),
                "Closeness Centrality": list(c_cent.values()),
                "Clustering Coefficient": [l_clust[n] for n in d_cent.keys()]
            })
            
            if node_labels:
                df_metrics["Node Name"] = df_metrics["Node"].map(node_labels)
            
            st.write("### 📊 Node Metrics Table (Centralities & Clustering)")
            st.dataframe(df_metrics.sort_values(by="Degree Centrality", ascending=False), use_container_width=True)
            
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