"""
utils.py - Data formatting, table export utilities, and deliverable packager.
"""

import sys
from pathlib import Path
from typing import Dict, Any
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import TABLES_DIR, PROCESSED_DATA_DIR


def export_all_deliverables_bundle(
    cleaned_df: pd.DataFrame,
    segmented_df: pd.DataFrame,
    kmeans_profiles_df: pd.DataFrame,
    hierarchical_profiles_df: pd.DataFrame,
    eval_metrics_df: pd.DataFrame,
    pca_variance_df: pd.DataFrame,
    pca_loadings_df: pd.DataFrame,
    dept_summary_df: pd.DataFrame,
    perf_summary_df: pd.DataFrame,
    retention_risk_df: pd.DataFrame,
    audit_log: Dict[str, Any],
) -> Dict[str, Path]:
    """
    Exports all required project data tables to CSV and Excel for reporting and auditing.
    """
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    exports = {}

    # 1. Cleaned Employee Dataset
    p1 = PROCESSED_DATA_DIR / "employee_cleaned.csv"
    cleaned_df.to_csv(p1, index=False)
    exports["cleaned_dataset"] = p1

    # 2. Employee-level Cluster Assignments
    p2 = PROCESSED_DATA_DIR / "employee_segmented.csv"
    segmented_df.to_csv(p2, index=False)
    exports["segmented_dataset"] = p2

    # 3. K-Means Cluster Profiles (CSV & Excel)
    p3 = TABLES_DIR / "kmeans_cluster_profiles.csv"
    kmeans_profiles_df.to_csv(p3, index=False)
    try:
        kmeans_profiles_df.to_excel(TABLES_DIR / "kmeans_cluster_profiles.xlsx", index=False)
    except Exception:
        pass
    exports["kmeans_profiles"] = p3

    # 4. Hierarchical Cluster Profiles
    p4 = TABLES_DIR / "hierarchical_cluster_profiles.csv"
    hierarchical_profiles_df.to_csv(p4, index=False)
    exports["hierarchical_profiles"] = p4

    # 5. Model Evaluation Metrics
    p5 = TABLES_DIR / "model_evaluation_metrics.csv"
    eval_metrics_df.to_csv(p5, index=False)
    exports["eval_metrics"] = p5

    # 6. PCA Explained Variance Table
    p6 = TABLES_DIR / "pca_explained_variance.csv"
    pca_variance_df.to_csv(p6, index=False)
    exports["pca_variance"] = p6

    # 7. PCA Component Loadings
    p7 = TABLES_DIR / "pca_component_loadings.csv"
    pca_loadings_df.to_csv(p7, index=False)
    exports["pca_loadings"] = p7

    # 8. Department-Level Segment Summary
    p8 = TABLES_DIR / "department_segment_summary.csv"
    dept_summary_df.to_csv(p8, index=False)
    exports["department_summary"] = p8

    # 9. Performance Summary
    p9 = TABLES_DIR / "performance_summary.csv"
    perf_summary_df.to_csv(p9, index=False)
    exports["performance_summary"] = p9

    # 10. Retention Risk Analysis
    p10 = TABLES_DIR / "retention_risk_analysis.csv"
    retention_risk_df.to_csv(p10, index=False)
    exports["retention_analysis"] = p10

    # 11. Data Quality & Audit Report
    audit_series = pd.DataFrame([audit_log]).T.reset_index()
    audit_series.columns = ["Audit_Metric", "Value"]
    p11 = TABLES_DIR / "data_quality_audit_report.csv"
    audit_series.to_csv(p11, index=False)
    exports["data_quality_audit"] = p11

    return exports
