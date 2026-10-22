"""
6_Reports_Downloads.py - Executive PDF report generation, deliverable export center, and data downloads.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    MAIN_REPORT_PDF,
    PCA_SUMMARY_PDF,
    TABLES_DIR,
    PROCESSED_DATA_DIR,
    DEFAULT_K_CLUSTERS,
    FAVICON_PATH,
    FLATICON_CDN_CSS,
)
from src.data_loader import load_or_generate_dataset
from src.data_cleaning import run_data_cleaning_pipeline
from src.feature_engineering import WorkforceFeaturePipeline
from src.clustering import train_kmeans_model, generate_cluster_profiles, evaluate_kmeans_candidates
from src.hierarchical_clustering import train_hierarchical_clustering, compare_clustering_algorithms
from src.pca_analysis import WorkforcePCAModel
from src.workforce_analysis import analyze_department_segments, compute_retention_risk_indicators, analyze_performance_and_satisfaction
from src.report_generator import generate_main_segmentation_report, generate_pca_summary_report
from src.utils import export_all_deliverables_bundle

st.set_page_config(page_title="Reports & Downloads | Workforce Segmentation", page_icon=str(FAVICON_PATH), layout="wide")
st.markdown(FLATICON_CDN_CSS, unsafe_allow_html=True)

st.markdown('<h2><i class="fi fi-sr-file-download header-icon"></i>Formal Reports & Data Export Center</h2>', unsafe_allow_html=True)
st.markdown("Generate and download comprehensive ReportLab PDF deliverables, analytical profiles, and data exports.")

# Generate/Cache full project bundle
@st.cache_data(show_spinner="Compiling data export bundle and verifying reports...")
def prepare_report_bundle():
    raw_df = load_or_generate_dataset()
    cleaned_df, audit_log = run_data_cleaning_pipeline(raw_df)

    feat_pipe = WorkforceFeaturePipeline()
    X_arr, _ = feat_pipe.fit_transform(cleaned_df)

    # Models
    _, km_labels, km_metrics = train_kmeans_model(X_arr, n_clusters=DEFAULT_K_CLUSTERS)
    _, h_labels, h_metrics = train_hierarchical_clustering(X_arr, n_clusters=DEFAULT_K_CLUSTERS)
    comp = compare_clustering_algorithms(km_labels, h_labels)

    eval_df = evaluate_kmeans_candidates(X_arr, k_range=(2, 6))

    profiles_df, _, names = generate_cluster_profiles(cleaned_df, km_labels)
    h_profiles_df, _, _ = generate_cluster_profiles(cleaned_df, h_labels, cluster_col_name="Hierarchical_Cluster")

    cleaned_df["KMeans_Cluster"] = km_labels
    cleaned_df["Hierarchical_Cluster"] = h_labels
    cleaned_df["Segment_Name"] = cleaned_df["KMeans_Cluster"].map(names)

    dept_res = analyze_department_segments(cleaned_df, segment_names=names)
    perf_res = analyze_performance_and_satisfaction(cleaned_df, segment_names=names)
    risk_df = compute_retention_risk_indicators(cleaned_df, segment_names=names)

    pca_model = WorkforcePCAModel()
    X_pca, pca_res = pca_model.fit_transform(X_arr, feat_pipe.transformed_feature_names)
    loadings_df = pca_model.get_loadings_dataframe(n_top_components=5)
    top_pc1 = pca_model.get_top_contributors(0, 6)
    top_pc2 = pca_model.get_top_contributors(1, 6)

    # Export all tabular deliverables
    export_all_deliverables_bundle(
        cleaned_df=cleaned_df,
        segmented_df=cleaned_df,
        kmeans_profiles_df=profiles_df,
        hierarchical_profiles_df=h_profiles_df,
        eval_metrics_df=eval_df,
        pca_variance_df=pca_res["variance_table"],
        pca_loadings_df=loadings_df,
        dept_summary_df=dept_res["department_summary"],
        perf_summary_df=perf_res["cluster_perf_summary"],
        retention_risk_df=risk_df,
        audit_log=audit_log,
    )

    # Summary pack for report
    summary_pack = {
        "total_records": len(cleaned_df),
        "ari": comp["adjusted_rand_index"],
        "nmi": comp["normalized_mutual_info"],
        "pc1_var": pca_res["variance_summary"]["pc1_explained_variance"],
        "pc2_var": pca_res["variance_summary"]["pc2_explained_variance"],
        "pc1_pc2_var": pca_res["variance_summary"]["pc1_pc2_cumulative_variance"],
        "profiles_df": profiles_df[["Cluster_ID", "Segment_Name", "Headcount", "Workforce_Share_Pct", "Avg_MonthlyIncome", "Avg_YearsAtCompany"]],
        "dept_df": dept_res["department_summary"][["Department", "Headcount", "Share_Pct", "Avg_Tenure_Years", "Avg_Monthly_Income"]],
        "risk_df": risk_df[["Cluster_ID", "Segment_Name", "Headcount", "Composite_Review_Index", "Review_Priority"]],
    }

    # Ensure PDFs exist
    if not MAIN_REPORT_PDF.exists():
        generate_main_segmentation_report(summary_pack, MAIN_REPORT_PDF)
    if not PCA_SUMMARY_PDF.exists():
        generate_pca_summary_report(pca_res, top_pc1, top_pc2, PCA_SUMMARY_PDF)

    return {
        "summary_pack": summary_pack,
        "pca_res": pca_res,
        "top_pc1": top_pc1,
        "top_pc2": top_pc2,
    }

bundle = prepare_report_bundle()

tab_pdf, tab_tables = st.tabs([
    "Formal PDF Deliverables",
    "Complete Tabular Data Exports (CSV / Excel)",
])

with tab_pdf:
    st.markdown('<h4><i class="fi fi-sr-document-signed card-icon"></i>Official PDF Deliverables (Programmatic ReportLab Generation)</h4>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<h4><i class="fi fi-sr-book-alt card-icon"></i>Main Project Report</h4>', unsafe_allow_html=True)
        st.markdown("**Title:** *Employee Workforce Segmentation Report*")
        st.markdown("""
        **Document Scope (21 Formal Sections):**
        - Executive Summary & Project Objectives
        - Dataset Description & Data Dictionary
        - Data Hygiene, Validation & Imputation Audit
        - Preprocessing, Scaling & One-Hot Encoding
        - K-Means Multi-Metric Evaluation (Inertia, Silhouette, Davies-Bouldin, Calinski-Harabasz)
        - Ward Hierarchical Clustering & SciPy Dendrogram
        - PCA Variance Scree & 2D Projection Space
        - Workforce Archetype Profiles & Distinguishing Attributes
        - Department-Level Cross-Tabulation & Composition
        - Retention Risk Diagnostics & Strategic HR Recommendations
        - Ethical Considerations & Responsible AI Mandate
        """)

        if MAIN_REPORT_PDF.exists():
            with open(MAIN_REPORT_PDF, "rb") as f:
                pdf_bytes = f.read()
            st.download_button(
                label="Download Main Segmentation Report (PDF)",
                data=pdf_bytes,
                file_name="employee_workforce_segmentation_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
            st.success(f"Generated PDF Size: {len(pdf_bytes) / 1024:.1f} KB")

    with col2:
        st.markdown('<h4><i class="fi fi-sr-chart-histogram card-icon"></i>PCA Summary Deliverable</h4>', unsafe_allow_html=True)
        st.markdown("**Title:** *PCA Summary — Employee Workforce Segmentation*")
        st.markdown("""
        **Document Scope:**
        - Mathematical Objective of Dimensionality Reduction
        - Input Matrix Preprocessing (35 features standardized)
        - Scree Plot & Cumulative Explained Variance Analysis
        - Eigenvalue Variance Thresholds (80%, 90%, 95%)
        - 2D Projection Subspace Verification
        - Leading Feature Loadings for PC1 and PC2
        - Information Loss Calculation (Residual Variance)
        - Technical Limitations & Manifold Recommendations
        """)

        if PCA_SUMMARY_PDF.exists():
            with open(PCA_SUMMARY_PDF, "rb") as f:
                pca_bytes = f.read()
            st.download_button(
                label="Download PCA Summary Report (PDF)",
                data=pca_bytes,
                file_name="pca_summary_workforce_segmentation.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
            st.success(f"Generated PDF Size: {len(pca_bytes) / 1024:.1f} KB")

    st.markdown("---")
    if st.button("Regenerate PDF Reports with Latest Data"):
        with st.spinner("Regenerating PDF documents..."):
            generate_main_segmentation_report(bundle["summary_pack"], MAIN_REPORT_PDF)
            generate_pca_summary_report(bundle["pca_res"], bundle["top_pc1"], bundle["top_pc2"], PCA_SUMMARY_PDF)
            st.success("Both PDF reports successfully recompiled!")

with tab_tables:
    st.markdown('<h4><i class="fi fi-sr-table card-icon"></i>Analytical Data Exports & Model Tables</h4>', unsafe_allow_html=True)
    st.markdown("Download all intermediate and final outputs in UTF-8 formatted CSV or Excel formats.")

    st.warning("**Data Privacy Notice:** Downloaded employee-level datasets contain organizational workforce records. Handle in accordance with internal employee privacy and compliance mandates.")

    export_files = [
        ("Cleaned Workforce Dataset", PROCESSED_DATA_DIR / "employee_cleaned.csv", "Cleaned, validated input records without identifiers."),
        ("Segmented Dataset (Cluster Labels)", PROCESSED_DATA_DIR / "employee_segmented.csv", "Full workforce records appended with K-Means & Hierarchical cluster IDs and segment names."),
        ("K-Means Cluster Profiles", TABLES_DIR / "kmeans_cluster_profiles.csv", "Mean statistics, workforce percentages, and names per K-Means cluster."),
        ("Hierarchical Cluster Profiles", TABLES_DIR / "hierarchical_cluster_profiles.csv", "Mean statistics and headcount distribution for Ward Hierarchical clusters."),
        ("Model Evaluation Metrics", TABLES_DIR / "model_evaluation_metrics.csv", "Inertia, Silhouette, Davies-Bouldin, and Calinski-Harabasz across candidate k values."),
        ("PCA Explained Variance Table", TABLES_DIR / "pca_explained_variance.csv", "Individual and cumulative explained variance ratios for all retained components."),
        ("PCA Component Loadings", TABLES_DIR / "pca_component_loadings.csv", "Correlation coefficients between original scaled features and leading principal components."),
        ("Department Segment Summary", TABLES_DIR / "department_segment_summary.csv", "Department headcount, compensation, tenure, satisfaction, and cluster composition."),
        ("Performance Analysis Summary", TABLES_DIR / "performance_summary.csv", "Performance rating distributions and high/low performer shares per segment."),
        ("Retention Risk Analysis", TABLES_DIR / "retention_risk_analysis.csv", "Composite friction index, promotion stagnation, overtime, and observed separation rates."),
        ("Data Quality Audit Report", TABLES_DIR / "data_quality_audit_report.csv", "Documentation of missing values, duplicate resolution, and logical corrections."),
    ]

    for title, path, desc in export_files:
        c_left, c_right = st.columns([3, 1])
        with c_left:
            st.markdown(f"**{title}**")
            st.caption(desc)
        with c_right:
            if path.exists():
                with open(path, "rb") as f:
                    file_data = f.read()
                st.download_button(
                    label="Download CSV",
                    data=file_data,
                    file_name=path.name,
                    mime="text/csv",
                    key=f"dl_{path.stem}",
                )
            else:
                st.info("File compiling...")
        st.divider()
