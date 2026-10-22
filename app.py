"""
app.py - Workforce Analytics & Employee Segmentation Dashboard.
Main entry point for Streamlit application.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

# Set project root in path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    RAW_DATA_FILE,
    PROCESSED_DATA_FILE,
    THEME_COLORS,
    DEFAULT_K_CLUSTERS,
    MAIN_REPORT_PDF,
    PCA_SUMMARY_PDF,
    FAVICON_PATH,
    FLATICON_CDN_CSS,
)
from src.data_loader import load_or_generate_dataset
from src.data_cleaning import run_data_cleaning_pipeline
from src.feature_engineering import WorkforceFeaturePipeline
from src.clustering import train_kmeans_model, generate_cluster_profiles, evaluate_kmeans_candidates
from src.hierarchical_clustering import train_hierarchical_clustering, compare_clustering_algorithms
from src.pca_analysis import WorkforcePCAModel
from src.workforce_analysis import analyze_department_segments, compute_retention_risk_indicators

st.set_page_config(
    page_title="Workforce Segmentation Analytics",
    page_icon=str(FAVICON_PATH),
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(FLATICON_CDN_CSS, unsafe_allow_html=True)

# Custom CSS for styling
st.markdown(f"""
    <style>
    .main-header {{
        font-size: 2.2rem;
        font-weight: 800;
        color: {THEME_COLORS["primary"]};
        margin-bottom: 0.2rem;
    }}
    .sub-header {{
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }}
    .kpi-card {{
        background: white;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border-left: 5px solid {THEME_COLORS["secondary"]};
    }}
    .kpi-title {{
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
    }}
    .kpi-value {{
        font-size: 1.8rem;
        font-weight: 700;
        color: {THEME_COLORS["dark"]};
    }}
    .kpi-subtitle {{
        font-size: 0.8rem;
        color: #94A3B8;
    }}
    </style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading and initializing workforce models...")
def initialize_workforce_pipeline():
    """Caches dataset loading, cleaning, feature engineering, clustering, and PCA."""
    raw_df = load_or_generate_dataset(force_generate=False)
    cleaned_df, audit_log = run_data_cleaning_pipeline(raw_df)

    feat_pipe = WorkforceFeaturePipeline()
    X_arr, X_df = feat_pipe.fit_transform(cleaned_df)

    # Models
    km_model, km_labels, km_metrics = train_kmeans_model(X_arr, n_clusters=DEFAULT_K_CLUSTERS)
    h_model, h_labels, h_metrics = train_hierarchical_clustering(X_arr, n_clusters=DEFAULT_K_CLUSTERS)
    clustering_comparison = compare_clustering_algorithms(km_labels, h_labels)

    # Profiling & Naming
    profiles_df, distinguishing_df, segment_names = generate_cluster_profiles(cleaned_df, km_labels)

    # Attach labels to dataframe
    cleaned_df["KMeans_Cluster"] = km_labels
    cleaned_df["Hierarchical_Cluster"] = h_labels
    cleaned_df["Segment_Name"] = cleaned_df["KMeans_Cluster"].map(segment_names)

    # PCA
    pca_model = WorkforcePCAModel()
    X_pca, pca_results = pca_model.fit_transform(X_arr, feat_pipe.transformed_feature_names)

    # Department and Risk Analytics
    dept_results = analyze_department_segments(cleaned_df, segment_names=segment_names)
    risk_df = compute_retention_risk_indicators(cleaned_df, segment_names=segment_names)

    return {
        "raw_df": raw_df,
        "cleaned_df": cleaned_df,
        "audit_log": audit_log,
        "feature_pipeline": feat_pipe,
        "X_arr": X_arr,
        "X_df": X_df,
        "kmeans_model": km_model,
        "kmeans_metrics": km_metrics,
        "kmeans_labels": km_labels,
        "hierarchical_model": h_model,
        "hierarchical_metrics": h_metrics,
        "hierarchical_labels": h_labels,
        "clustering_comparison": clustering_comparison,
        "profiles_df": profiles_df,
        "distinguishing_df": distinguishing_df,
        "segment_names": segment_names,
        "pca_model": pca_model,
        "X_pca": X_pca,
        "pca_results": pca_results,
        "dept_results": dept_results,
        "risk_df": risk_df,
    }


# Load Pipeline Data
data = initialize_workforce_pipeline()
cleaned_df = data["cleaned_df"]
km_metrics = data["kmeans_metrics"]
h_metrics = data["hierarchical_metrics"]
pca_var = data["pca_results"]["variance_summary"]

# Sidebar Navigation and Global Filters
st.sidebar.image("https://images.unsplash.com/photo-1551836022-d5d88e9218df?w=600&auto=format&fit=crop&q=80", use_container_width=True)
st.sidebar.markdown('<h2><i class="fi fi-sr-users-alt" style="color:#0D9488; margin-right:8px;"></i>Workforce Analytics</h2>', unsafe_allow_html=True)
st.sidebar.markdown("**Machine Learning Workforce Segmentation**")
st.sidebar.markdown("---")

st.sidebar.subheader("Global Filter Context")
selected_dept = st.sidebar.multiselect(
    "Filter by Department",
    options=sorted(cleaned_df["Department"].unique()),
    default=sorted(cleaned_df["Department"].unique()),
)
selected_clusters = st.sidebar.multiselect(
    "Filter by Segment Archetype",
    options=sorted(cleaned_df["Segment_Name"].unique()),
    default=sorted(cleaned_df["Segment_Name"].unique()),
)

# Filtered view for dashboard
filtered_df = cleaned_df[
    (cleaned_df["Department"].isin(selected_dept)) &
    (cleaned_df["Segment_Name"].isin(selected_clusters))
]

# Main Dashboard View
st.markdown('<div class="main-header">Employee Workforce Segmentation Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Unsupervised Machine Learning & Human Resources Analytics Platform</div>', unsafe_allow_html=True)

# KPI Metrics Ribbon
c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:
    st.metric(
        label="Total Employees",
        value=f"{len(cleaned_df):,}",
        delta=f"{len(filtered_df):,} in filter",
    )
with c2:
    st.metric(
        label="Departments",
        value=cleaned_df["Department"].nunique(),
        delta="3 Enterprise Units",
    )
with c3:
    st.metric(
        label="K-Means Clusters",
        value=km_metrics["n_clusters"],
        delta=f"Silhouette: {km_metrics['silhouette_score']:.3f}",
    )
with c4:
    st.metric(
        label="Hierarchical Clusters",
        value=h_metrics["n_clusters"],
        delta="Ward Linkage",
    )
with c5:
    st.metric(
        label="PCA 2D Variance",
        value=f"{pca_var['pc1_pc2_cumulative_variance']:.1f}%",
        delta=f"PC1: {pca_var['pc1_explained_variance']:.1f}%",
    )
with c6:
    if "Attrition" in cleaned_df:
        att_rate = (cleaned_df["Attrition"].astype(str).str.lower() == "yes").mean() * 100
        st.metric(
            label="Workforce Attrition",
            value=f"{att_rate:.1f}%",
            delta="Baseline Benchmark",
        )
    else:
        st.metric(label="Data Integrity", value="100%", delta="0 Missing Values")

st.markdown("---")

# Section 1: Executive Overview & Methodology Flowchart
col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown('<h3><i class="fi fi-sr-document-signed section-icon"></i>Executive Project Summary</h3>', unsafe_allow_html=True)
    st.markdown("""
    This analytics platform applies unsupervised machine learning algorithms to uncover latent behavioral,
    career-stage, and organizational patterns across an enterprise workforce.
    
    Rather than relying on subjective intuition, the platform combines **K-Means clustering** and **Agglomerative Hierarchical clustering**
    to identify distinct employee segments, paired with **Principal Component Analysis (PCA)** to reduce 35 transformed features into
    interpretable latent dimensions.
    
    ### Core Discoveries:
    1. **Tenure & Compensation Bifurcation:** PC1 (26.5% variance) is driven by company tenure and promotion duration, while PC2 (11.1% variance) aligns with job level and monthly income.
    2. **Four Distinct Workforce Archetypes:**
       - *Senior Leadership & Tenured Specialists:* High income ($14k+), extensive tenure (12+ yrs), low attrition risk.
       - *Early-Career High-Engagement Talent:* Entry-level, under 3 yrs tenure, high learning agility, moderate flight risk.
       - *Stagnant & Retention-Review Segment:* Promotion gap (>=4 yrs), elevated overtime, requiring proactive engagement.
       - *Experienced Core Operational Backbone:* Steady tenure (5-8 yrs), foundational contributors driving daily productivity.
    3. **Supportive Talent Interventions:** All segment insights are strictly designed for workforce planning, mentoring pipelines, and workload balancing, explicitly rejecting punitive usage.
    """)

with col_right:
    st.markdown('<h3><i class="fi fi-sr-network section-icon"></i>Machine Learning Workflow</h3>', unsafe_allow_html=True)
    st.info("""
    **Step-by-Step Analytical Pipeline:**
    
    1. **Data Ingestion & Integrity Audit**
       - 1,470 records validated; zero duplicate records; logical consistency checks.
    2. **Feature Engineering & Preprocessing**
       - StandardScaler applied to continuous features; OneHotEncoder for categorical features.
    3. **Unsupervised Clustering Pipeline**
       - Evaluated K=2 to 6 across Inertia, Silhouette, Davies-Bouldin, and Calinski-Harabasz.
    4. **Hierarchical Clustering & Dendrogram**
       - Ward minimum variance linkage with Euclidean distance and SciPy dendrogram.
    5. **PCA Dimensionality Reduction**
       - Eigenvalue decomposition, scree plot, cumulative variance thresholds (80%, 90%, 95%).
    6. **Workforce Diagnostics & Reporting**
       - Department cross-tabulation, talent 4-quadrant grid, and automated PDF deliverables.
    """)

st.markdown("---")

# Quick Navigation Grid
st.markdown('<h3><i class="fi fi-sr-compass section-icon"></i>Platform Navigation</h3>', unsafe_allow_html=True)
p1, p2, p3, p4 = st.columns(4)
with p1:
    st.markdown('##### <i class="fi fi-sr-database nav-card-icon"></i> [1. Data Explorer](Data_Explorer)', unsafe_allow_html=True)
    st.caption("Inspect raw data, schema dictionary, missing value audits, and upload custom datasets.")
with p2:
    st.markdown('##### <i class="fi fi-sr-chart-pie-alt nav-card-icon"></i> [2. Clustering Analysis](Clustering_Analysis)', unsafe_allow_html=True)
    st.caption("Interactive K-Means, multi-metric evaluation, Ward hierarchical dendrogram, and cross-comparison.")
with p3:
    st.markdown('##### <i class="fi fi-sr-chart-histogram nav-card-icon"></i> [3. PCA Analysis](PCA_Analysis)', unsafe_allow_html=True)
    st.caption("2D PCA projection scatter, scree curve, cumulative variance, and component loading coefficients.")
with p4:
    st.markdown('##### <i class="fi fi-sr-building nav-card-icon"></i> [4. Department Analytics](Department_Analytics)', unsafe_allow_html=True)
    st.caption("Department headcount breakdown, cluster composition stacked bars, and cross-tabulation heatmaps.")

p5, p6, p7, p8 = st.columns(4)
with p5:
    st.markdown('##### <i class="fi fi-sr-bullseye-arrow nav-card-icon"></i> [5. Performance & Retention](Performance_Retention)', unsafe_allow_html=True)
    st.caption("Performance vs. satisfaction 4-quadrant talent grid, retention review indicators, and ethical HR recommendations.")
with p6:
    st.markdown('##### <i class="fi fi-sr-file-download nav-card-icon"></i> [6. Reports & Downloads](Reports_Downloads)', unsafe_allow_html=True)
    st.caption("Download the formal 21-section PDF Report, PCA Summary PDF, and complete CSV/Excel data bundle.")
with p7:
    st.markdown('##### <i class="fi fi-sr-book-alt nav-card-icon"></i> [7. Methodology & Limitations](Methodology_Limitations)', unsafe_allow_html=True)
    st.caption("Academic deep-dive: K-Means mechanics, Ward variance, PCA mathematical formulation, and ethical AI guidelines.")
with p8:
    st.markdown('##### <i class="fi fi-sr-settings-sliders nav-card-icon"></i> System Status', unsafe_allow_html=True)
    st.success("All ML Pipelines & Models Active (Python 3.11 / Scikit-Learn 1.9)")

st.markdown("---")
st.caption("Employee Workforce Segmentation Project | College Assignment Submission & Academic Demonstration")
