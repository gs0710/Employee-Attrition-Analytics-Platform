"""
3_PCA_Analysis.py - Principal Component Analysis, variance decomposition, loadings, and 2D projections.
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
from src.clustering import train_kmeans_model, generate_cluster_profiles
from src.hierarchical_clustering import train_hierarchical_clustering
from src.pca_analysis import WorkforcePCAModel
from src.visualization import (
    create_pca_variance_plot,
    create_pca_2d_scatter_plot,
    create_loadings_bar_chart,
)

st.set_page_config(page_title="PCA Analysis | Workforce Segmentation", page_icon=str(FAVICON_PATH), layout="wide")
st.markdown(FLATICON_CDN_CSS, unsafe_allow_html=True)

st.markdown('<h2><i class="fi fi-sr-chart-histogram header-icon"></i>Principal Component Analysis (PCA)</h2>', unsafe_allow_html=True)
st.markdown("Dimensionality reduction, scree analysis, leading feature loadings, and 2D workforce projections.")

# Cached PCA pipeline
@st.cache_data
def get_pca_data():
    raw_df = load_or_generate_dataset()
    cleaned_df, _ = run_data_cleaning_pipeline(raw_df)

    feat_pipe = WorkforceFeaturePipeline()
    X_arr, _ = feat_pipe.fit_transform(cleaned_df)

    # Models for coloring
    _, km_labels, _ = train_kmeans_model(X_arr, n_clusters=DEFAULT_K_CLUSTERS)
    _, h_labels, _ = train_hierarchical_clustering(X_arr, n_clusters=DEFAULT_K_CLUSTERS)
    _, _, names = generate_cluster_profiles(cleaned_df, km_labels)

    cleaned_df["KMeans_Cluster"] = [names.get(i, f"Cluster {i}") for i in km_labels]
    cleaned_df["Hierarchical_Cluster"] = [f"H-Cluster {i}" for i in h_labels]

    pca_model = WorkforcePCAModel()
    X_pca, pca_results = pca_model.fit_transform(X_arr, feat_pipe.transformed_feature_names)
    loadings_df = pca_model.get_loadings_dataframe(n_top_components=5)

    return cleaned_df, X_pca, pca_results, pca_model, loadings_df

cleaned_df, X_pca, pca_results, pca_model, loadings_df = get_pca_data()
var_sum = pca_results["variance_summary"]
var_df = pca_results["variance_table"]

# Variance KPI Ribbon
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Original Features", var_sum["total_input_features"])
k2.metric("PC1 Variance", f"{var_sum['pc1_explained_variance']:.2f}%", delta="Career Longevity")
k3.metric("PC2 Variance", f"{var_sum['pc2_explained_variance']:.2f}%", delta="Seniority & Income")
k4.metric("PC1 + PC2 Combined", f"{var_sum['pc1_pc2_cumulative_variance']:.2f}%", delta="2D Plane")
k5.metric("80% Variance Threshold", f"{var_sum['components_for_80_pct']} PCs", delta="of 35 dimensions")

st.markdown("---")

tab_scatter, tab_scree, tab_loadings, tab_theory = st.tabs([
    "2D Workforce Projection Space",
    "Scree Plot & Cumulative Variance",
    "Component Feature Loadings",
    "Technical Nuance & Information Loss",
])

with tab_scatter:
    st.markdown('<h4><i class="fi fi-sr-chart-network card-icon"></i>Interactive 2D Principal Component Space</h4>', unsafe_allow_html=True)

    col_ctrl, col_plot = st.columns([1, 4])
    with col_ctrl:
        st.markdown("##### Projection Controls")
        color_dimension = st.selectbox(
            "Color Points By:",
            [
                "KMeans_Cluster",
                "Hierarchical_Cluster",
                "Department",
                "JobRole",
                "JobSatisfaction",
                "PerformanceRating",
                "Attrition",
                "Overtime",
            ],
            index=0,
        )
        st.caption("Hover over individual points to inspect employee records.")

    with col_plot:
        scatter_fig = create_pca_2d_scatter_plot(
            cleaned_df,
            X_pca,
            color_col=color_dimension,
            title=f"Workforce Projection colored by {color_dimension}",
            pc1_var=var_sum["pc1_explained_variance"],
            pc2_var=var_sum["pc2_explained_variance"],
        )
        st.plotly_chart(scatter_fig, use_container_width=True)

with tab_scree:
    st.markdown('<h4><i class="fi fi-sr-chart-histogram card-icon"></i>Eigenvalue Scree Plot & Variance Accumulation</h4>', unsafe_allow_html=True)
    scree_fig = create_pca_variance_plot(var_df, max_components=15)
    st.plotly_chart(scree_fig, use_container_width=True)

    with st.expander("View Explained Variance Table (All Retained Components)", expanded=False):
        st.dataframe(var_df.style.format({
            "Explained_Variance_Ratio": "{:.4f}",
            "Explained_Variance_Pct": "{:.2f}%",
            "Cumulative_Variance_Pct": "{:.2f}%",
        }), use_container_width=True)

with tab_loadings:
    st.markdown('<h4><i class="fi fi-sr-scale-comparison-alt card-icon"></i>Feature Loadings (Component Deconstruction)</h4>', unsafe_allow_html=True)
    st.markdown("Discover the specific workforce variables that construct each principal axis.")

    c_sel, c_view = st.columns([1, 3])
    with c_sel:
        selected_pc = st.selectbox("Select Component to Deconstruct:", ["PC1", "PC2", "PC3", "PC4", "PC5"], index=0)
        pc_idx = int(selected_pc.replace("PC", "")) - 1
        top_n = st.slider("Top Contributing Features:", 4, 12, 8)

    top_df = pca_model.get_top_contributors(component_idx=pc_idx, top_n=top_n)

    with c_view:
        loadings_fig = create_loadings_bar_chart(top_df, component_name=selected_pc)
        st.plotly_chart(loadings_fig, use_container_width=True)

    st.markdown("##### Detailed Loading Coefficients Matrix")
    st.dataframe(loadings_df.style.background_gradient(cmap="coolwarm", axis=0).format("{:.4f}"), use_container_width=True)

with tab_theory:
    st.markdown('<h4><i class="fi fi-sr-document-signed card-icon"></i>Information Loss & Methodological Considerations</h4>', unsafe_allow_html=True)
    st.markdown(f"""
    ### 1. Variance Compression & Information Loss
    - **Total Variance Retained:** The 2D PCA scatter plot retains **{var_sum['pc1_pc2_cumulative_variance']:.2f}%** of total feature variance.
    - **Residual Dimensions:** Approximately **{100 - var_sum['pc1_pc2_cumulative_variance']:.2f}%** of workforce variation resides in orthogonal dimensions PC3 through PC35.
    - **Clustering Implication:** For this critical reason, our clustering algorithms (K-Means and Hierarchical Clustering) are fitted on the **entire 35-dimensional standardized space**, rather than prematurely truncating data to 2 components. PCA is utilized primarily for structural validation and visualization.

    ### 2. Semantic Interpretation of Axes
    - **PC1 (Horizontal Axis):** Driven predominantly by `YearsAtCompany` (+0.44), `TotalWorkingYears` (+0.42), and `YearsSinceLastPromotion` (+0.37). It functions as an **Organizational Longevity & Tenure Index**.
    - **PC2 (Vertical Axis):** Dominated by `JobLevel` (+0.66) and `MonthlyIncome` (+0.63). It operates as an **Executive Grade & Compensation Index**.
    """)
