"""
test_reports.py - Unit tests for ReportLab PDF report generation and bundle exports.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.report_generator import generate_main_segmentation_report, generate_pca_summary_report


def test_main_segmentation_report_generation(tmp_path):
    """Verifies that the main PDF segmentation report compiles programmatically."""
    test_pdf = tmp_path / "test_main_report.pdf"

    mock_summary = {
        "total_records": 100,
        "ari": 0.35,
        "nmi": 0.42,
        "pc1_var": 25.0,
        "pc2_var": 12.0,
        "pc1_pc2_var": 37.0,
        "profiles_df": pd.DataFrame({
            "Cluster_ID": [0, 1],
            "Segment_Name": ["Group A", "Group B"],
            "Headcount": [60, 40],
            "Workforce_Share_Pct": [60.0, 40.0],
            "Avg_MonthlyIncome": [5000, 8000],
            "Avg_YearsAtCompany": [3.5, 7.2],
        }),
        "dept_df": pd.DataFrame({
            "Department": ["Sales", "R&D"],
            "Headcount": [40, 60],
            "Share_Pct": [40.0, 60.0],
            "Avg_Tenure_Years": [4.0, 6.0],
            "Avg_Monthly_Income": [5500, 7500],
        }),
        "risk_df": pd.DataFrame({
            "Cluster_ID": [0, 1],
            "Segment_Name": ["Group A", "Group B"],
            "Headcount": [60, 40],
            "Composite_Review_Index": [30.0, 15.0],
            "Review_Priority": ["Moderate Attention", "Stable Engagement"],
        }),
    }

    out_path = generate_main_segmentation_report(mock_summary, pdf_path=test_pdf)
    assert out_path.exists()
    assert out_path.stat().st_size > 1000  # Valid non-empty PDF


def test_pca_summary_report_generation(tmp_path):
    """Verifies that the standalone PCA summary PDF compiles programmatically."""
    test_pdf = tmp_path / "test_pca_summary.pdf"

    mock_pca_results = {
        "variance_summary": {
            "total_input_features": 30,
            "total_components": 30,
            "pc1_explained_variance": 26.5,
            "pc2_explained_variance": 11.2,
            "pc1_pc2_cumulative_variance": 37.7,
            "components_for_80_pct": 10,
            "components_for_90_pct": 15,
            "components_for_95_pct": 19,
        }
    }
    mock_top_pc1 = pd.DataFrame({
        "Feature": ["Tenure", "Experience", "Promotion"],
        "Loading": [0.45, 0.40, 0.35],
        "Absolute_Loading": [0.45, 0.40, 0.35],
        "Direction": ["Positive", "Positive", "Positive"],
    })
    mock_top_pc2 = pd.DataFrame({
        "Feature": ["JobLevel", "Income", "Role"],
        "Loading": [0.65, 0.60, -0.20],
        "Absolute_Loading": [0.65, 0.60, 0.20],
        "Direction": ["Positive", "Positive", "Negative"],
    })

    out_path = generate_pca_summary_report(mock_pca_results, mock_top_pc1, mock_top_pc2, pdf_path=test_pdf)
    assert out_path.exists()
    assert out_path.stat().st_size > 1000
