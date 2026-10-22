"""
2_Clustering_Analysis.py - Interactive K-Means and Hierarchical Clustering, dendrograms, and model comparison.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DEFAULT_K_CLUSTERS, THEME_COLORS, FAVICON_PATH, FLATICON_CDN_CSS
from src.data_loader import load_or_generate_dataset
from src.data_cleaning import run_data_cleaning_pipeline
from src.feature_engineering import WorkforceFeaturePipeline
from src.clustering import evaluate_kmeans_candidates, train_kmeans_model, generate_cluster_profiles
from src.hierarchical_clustering import (
    train_hierarchical_clustering,
    compute_scipy_linkage,
    compare_clustering_algorithms,
)
from src.visualization import (
    create_clustering_evaluation_plot,
    generate_dendrogram_plot,
)

st.set_page_config(page_title="Clustering Analysis | Workforce Segmentation", page_icon=str(FAVICON_PATH), layout="wide")
st.markdown(FLATICON_CDN_CSS, unsafe_allow_html=True)

st.markdown('<h2><i class="fi fi-sr-chart-pie-alt header-icon"></i>Unsupervised Workforce Clustering</h2>', unsafe_allow_html=True)
st.markdown("Discover latent workforce archetypes through K-Means and Agglomerative Hierarchical Clustering.")

# Cached feature matrix
@st.cache_data
def get_processed_data():
    raw_df = load_or_generate_dataset()
    cleaned_df, _ = run_data_cleaning_pipeline(raw_df)
    feat_pipe = WorkforceFeaturePipeline()
    X_arr, _ = feat_pipe.fit_transform(cleaned_df)
    return cleaned_df, X_arr, feat_pipe.transformed_feature_names

cleaned_df, X_arr, feat_names = get_processed_data()

tab_kmeans, tab_hierarchical, tab_compare = st.tabs([
    "K-Means Clustering",
    "Hierarchical Clustering & Dendrogram",
    "Algorithm Cross-Comparison",
])

with tab_kmeans:
    st.markdown('<h4><i class="fi fi-sr-bolt card-icon"></i>K-Means Model Execution & Multi-Metric Evaluation</h4>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 3])
    with c1:
        st.markdown("##### Configuration")
        k_val = st.slider("Select Number of Clusters (k):", min_value=2, max_value=8, value=DEFAULT_K_CLUSTERS)
        n_init = st.selectbox("Initialization Trials (n_init):", [10, 20, 30], index=1)
        st.caption("Random Seed: 42 (Enforces strict academic reproducibility)")

    # Candidate evaluation table and plot
    eval_df = evaluate_kmeans_candidates(X_arr, k_range=(2, 7))

    with c2:
        eval_fig = create_clustering_evaluation_plot(eval_df)
        st.plotly_chart(eval_fig, use_container_width=True)

    with st.expander("View Complete Evaluation Metrics Table (k=2 to 7)", expanded=False):
        st.dataframe(eval_df.style.highlight_max(subset=["Silhouette_Score", "Calinski_Harabasz_Index"], color="#D1FAE5")
                                 .highlight_min(subset=["Inertia", "Davies_Bouldin_Index"], color="#E0E7FF"),
                     use_container_width=True)

    # Train current selected k
    km_model, km_labels, km_metrics = train_kmeans_model(X_arr, n_clusters=k_val)
    profiles_df, distinguishing_df, segment_names = generate_cluster_profiles(cleaned_df, km_labels)

    st.markdown("---")
    st.subheader(f"Identified Workforce Segments (k = {k_val})")

    # Render Segment Profiles
    st.dataframe(profiles_df.style.format({
        "Workforce_Share_Pct": "{:.1f}%",
        "Avg_MonthlyIncome": "${:,.0f}",
        "Avg_Age": "{:.1f} yrs",
        "Avg_YearsAtCompany": "{:.1f} yrs",
        "Avg_TotalWorkingYears": "{:.1f} yrs",
        "Avg_YearsSinceLastPromotion": "{:.1f} yrs",
        "Avg_JobSatisfaction": "{:.2f}",
        "Avg_PerformanceRating": "{:.2f}",
        "Observed_Attrition_Pct": "{:.1f}%",
        "Overtime_Rate_Pct": "{:.1f}%",
    }), use_container_width=True)

    # Segment Cards
    st.markdown("##### Segment Archetype Profiles")
    cols = st.columns(k_val)
    for idx, col in enumerate(cols):
        with col:
            s_name = segment_names.get(idx, f"Cluster {idx}")
            s_row = profiles_df[profiles_df["Cluster_ID"] == idx].iloc[0]
            st.markdown(f"**Cluster {idx}**")
            inc_val = s_row.get('Avg_MonthlyIncome', 0)
            ten_val = s_row.get('Avg_YearsAtCompany', 0)
            sat_val = s_row.get('Avg_JobSatisfaction', s_row.get('Avg_Job_Satisfaction', 0))
            att_val = s_row.get('Observed_Attrition_Pct', 0)
            st.info(f"**{s_name}**\n\n"
                    f"• Headcount: **{s_row.get('Headcount', 0)}** ({s_row.get('Workforce_Share_Pct', 0):.1f}%)\n"
                    f"• Avg Income: **${inc_val:,.0f}**\n"
                    f"• Avg Tenure: **{ten_val:.1f} yrs**\n"
                    f"• Satisfaction: **{sat_val:.2f} / 4**\n"
                    f"• Observed Attrition: **{att_val:.1f}%**")

with tab_hierarchical:
    st.markdown('<h4><i class="fi fi-sr-network card-icon"></i>Agglomerative Hierarchical Clustering (Ward Linkage)</h4>', unsafe_allow_html=True)
    st.markdown("""
    Hierarchical clustering establishes a nested hierarchy of clusters using a bottom-up agglomerative approach.
    At each iteration, the algorithm merges the pair of clusters that minimizes the increase in total within-cluster variance (**Ward's criterion**).
    """)

    col_h1, col_h2 = st.columns([1, 2])
    with col_h1:
        h_k = st.slider("Select Hierarchical Clusters:", min_value=2, max_value=6, value=DEFAULT_K_CLUSTERS)
        sample_size = st.slider("Dendrogram Sample Size (N):", min_value=30, max_value=100, value=60, step=10)
        st.caption("Subsampling is applied exclusively to dendrogram rendering for visual clarity; clustering models assign all employees.")

    # Train Hierarchical
    h_model, h_labels, h_metrics = train_hierarchical_clustering(X_arr, n_clusters=h_k)
    h_profiles, _, h_names = generate_cluster_profiles(cleaned_df, h_labels, cluster_col_name="Hierarchical_Cluster")

    with col_h2:
        m1, m2, m3 = st.columns(3)
        m1.metric("Silhouette Score", f"{h_metrics['silhouette_score']:.4f}")
        m2.metric("Davies-Bouldin Index", f"{h_metrics['davies_bouldin_index']:.4f}")
        m3.metric("Calinski-Harabasz", f"{h_metrics['calinski_harabasz_index']:.1f}")

    # SciPy Dendrogram
    st.markdown("##### SciPy Dendrogram Visualization")
    Z, sampled_idx = compute_scipy_linkage(X_arr, sample_size=sample_size)
    dendro_path = generate_dendrogram_plot(Z, sample_size=sample_size)
    st.image(str(dendro_path), caption=f"Agglomerative Dendrogram (Ward Linkage, N={sample_size} sampled employees)", use_container_width=True)

    st.markdown("##### Hierarchical Cluster Profiles")
    st.dataframe(h_profiles.style.format({
        "Workforce_Share_Pct": "{:.1f}%",
        "Avg_MonthlyIncome": "${:,.0f}",
        "Avg_YearsAtCompany": "{:.1f} yrs",
        "Avg_JobSatisfaction": "{:.2f}",
        "Observed_Attrition_Pct": "{:.1f}%",
    }), use_container_width=True)

with tab_compare:
    st.markdown('<h4><i class="fi fi-sr-scale-comparison-alt card-icon"></i>Algorithm Comparison: K-Means vs. Agglomerative Hierarchical</h4>', unsafe_allow_html=True)

    comp_res = compare_clustering_algorithms(km_labels, h_labels)

    ca1, ca2 = st.columns(2)
    with ca1:
        st.metric("Adjusted Rand Index (ARI)", f"{comp_res['adjusted_rand_index']:.4f}")
        st.caption("Measures cluster agreement corrected for chance (0 = random, 1 = identical).")
    with ca2:
        st.metric("Normalized Mutual Information (NMI)", f"{comp_res['normalized_mutual_info']:.4f}")
        st.caption("Information-theoretic concordance between partition structures (0 to 1).")

    st.markdown("##### Contingency Cross-Tabulation Matrix")
    st.dataframe(comp_res["contingency_matrix"], use_container_width=True)

    st.info(f"""
    **Comparative Architectural Findings:**
    
    1. **Centroid-Based vs. Tree-Based Merging:** K-Means iteratively minimizes distance to Euclidean centroids, producing spherical Voronoi partitions. In contrast, Agglomerative Hierarchical builds an irreversible binary tree hierarchy.
    2. **Concordance Level:** The Adjusted Rand Index of **{comp_res['adjusted_rand_index']:.4f}** and NMI of **{comp_res['normalized_mutual_info']:.4f}** indicate solid alignment on core workforce archetypes (particularly isolating senior leadership and early-career cohorts), with nuanced differences along the boundary of mid-tenure contributors.
    """)
