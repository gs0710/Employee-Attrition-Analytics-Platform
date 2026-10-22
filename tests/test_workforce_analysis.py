"""
test_workforce_analysis.py - Unit tests for department segmentation, performance grids, and retention indicators.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import generate_synthetic_workforce_data
from src.data_cleaning import run_data_cleaning_pipeline
from src.workforce_analysis import (
    analyze_department_segments,
    analyze_performance_and_satisfaction,
    compute_retention_risk_indicators,
    generate_hr_recommendations,
)
from src.clustering import generate_cluster_profiles


@pytest.fixture
def segmented_test_dataframe():
    df = generate_synthetic_workforce_data(n_samples=100, random_seed=42)
    cleaned, _ = run_data_cleaning_pipeline(df)
    # Assign mock clusters
    cleaned["KMeans_Cluster"] = np.random.choice([0, 1, 2], size=len(cleaned))
    names = {0: "Segment A", 1: "Segment B", 2: "Segment C"}
    cleaned["Segment_Name"] = cleaned["KMeans_Cluster"].map(names)
    return cleaned, names


def test_department_segment_analytics(segmented_test_dataframe):
    """Verifies department breakdown, contingency matrix, and summary statistics."""
    df, names = segmented_test_dataframe
    dept_res = analyze_department_segments(df, segment_names=names)

    summary = dept_res["department_summary"]
    assert len(summary) == df["Department"].nunique()
    assert "Headcount" in summary.columns
    assert "Avg_Monthly_Income" in summary.columns
    assert "Avg_Tenure_Years" in summary.columns
    assert summary["Headcount"].sum() == len(df)


def test_performance_and_satisfaction_grid(segmented_test_dataframe):
    """Verifies the 4-quadrant strategic talent grid."""
    df, names = segmented_test_dataframe
    perf_res = analyze_performance_and_satisfaction(df, segment_names=names)

    q_df = perf_res["quadrant_distribution"]
    assert len(q_df) <= 4
    assert np.isclose(q_df["Workforce_Pct"].sum(), 100.0, atol=1.0)
    assert "Talent_Quadrant" in perf_res["df_with_quadrants"].columns


def test_retention_risk_indicator_calculation(segmented_test_dataframe):
    """Verifies retention friction indicator index and review priority categorization."""
    df, names = segmented_test_dataframe
    risk_df = compute_retention_risk_indicators(df, segment_names=names)

    assert len(risk_df) == 3
    assert "Composite_Review_Index" in risk_df.columns
    assert "Review_Priority" in risk_df.columns
    assert "Low_Job_Satisfaction_Pct" in risk_df.columns
    assert "Promotion_Stagnant_Pct" in risk_df.columns


def test_missing_attrition_column_graceful_handling():
    """Verifies that workforce analytics functions execute without error even if Attrition is missing."""
    df = generate_synthetic_workforce_data(n_samples=50, random_seed=42)
    df = df.drop(columns=["Attrition"], errors="ignore")
    cleaned, _ = run_data_cleaning_pipeline(df)
    cleaned["KMeans_Cluster"] = np.random.choice([0, 1], size=len(cleaned))
    names = {0: "Mock A", 1: "Mock B"}

    dept_res = analyze_department_segments(cleaned, segment_names=names)
    assert len(dept_res["department_summary"]) == cleaned["Department"].nunique()

    risk_df = compute_retention_risk_indicators(cleaned, segment_names=names)
    assert len(risk_df) == 2
    assert "Composite_Review_Index" in risk_df.columns
