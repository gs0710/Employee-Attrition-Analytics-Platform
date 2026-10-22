"""
eda.py - Exploratory Data Analysis computations and statistical summaries.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    NUMERICAL_FEATURES,
    DEPARTMENT_COLUMN,
    JOB_ROLE_COLUMN,
    ATTRITION_COLUMN,
    PERFORMANCE_COLUMN,
)


def generate_eda_summary_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes comprehensive EDA statistics:
    - Numerical descriptive statistics (mean, std, min, 25%, 50%, 75%, max, skewness)
    - Department & Job Role counts and percentages
    - Satisfaction & Performance distributions
    - Attrition summary if available
    - Correlation matrix of numerical variables
    """
    summary = {}

    # Numerical description
    avail_num = [c for c in NUMERICAL_FEATURES if c in df.columns]
    num_stats = df[avail_num].describe().round(2).T
    # Add skewness
    num_stats["skewness"] = df[avail_num].skew().round(2)
    summary["numerical_statistics"] = num_stats

    # Department distribution
    if DEPARTMENT_COLUMN in df.columns:
        dept_counts = df[DEPARTMENT_COLUMN].value_counts().reset_index()
        dept_counts.columns = ["Department", "Count"]
        dept_counts["Percentage"] = (dept_counts["Count"] / len(df) * 100).round(2)
        summary["department_distribution"] = dept_counts

    # Job Role distribution
    if JOB_ROLE_COLUMN in df.columns:
        role_counts = df[JOB_ROLE_COLUMN].value_counts().reset_index()
        role_counts.columns = ["JobRole", "Count"]
        role_counts["Percentage"] = (role_counts["Count"] / len(df) * 100).round(2)
        summary["job_role_distribution"] = role_counts

    # Performance distribution
    if PERFORMANCE_COLUMN in df.columns:
        perf_counts = df[PERFORMANCE_COLUMN].value_counts(sort=False).reset_index()
        perf_counts.columns = ["PerformanceRating", "Count"]
        perf_counts["Percentage"] = (perf_counts["Count"] / len(df) * 100).round(2)
        summary["performance_distribution"] = perf_counts

    # Job Satisfaction distribution
    if "JobSatisfaction" in df.columns:
        sat_counts = df["JobSatisfaction"].value_counts(sort=False).reset_index()
        sat_counts.columns = ["JobSatisfaction", "Count"]
        sat_counts["Percentage"] = (sat_counts["Count"] / len(df) * 100).round(2)
        summary["satisfaction_distribution"] = sat_counts

    # Attrition distribution
    if ATTRITION_COLUMN in df.columns:
        att_counts = df[ATTRITION_COLUMN].value_counts().reset_index()
        att_counts.columns = ["Attrition", "Count"]
        att_counts["Percentage"] = (att_counts["Count"] / len(df) * 100).round(2)
        summary["attrition_distribution"] = att_counts

    # Correlation Matrix
    if len(avail_num) > 1:
        corr_matrix = df[avail_num].corr().round(3)
        summary["correlation_matrix"] = corr_matrix

    # Missing values summary
    missing = df.isnull().sum()
    missing_df = pd.DataFrame({
        "Column": missing.index,
        "Missing_Count": missing.values,
        "Missing_Pct": np.round(missing.values / len(df) * 100, 2),
    })
    summary["missing_values"] = missing_df

    return summary


if __name__ == "__main__":
    from src.data_loader import load_or_generate_dataset
    df = load_or_generate_dataset()
    eda_stats = generate_eda_summary_statistics(df)
    print("EDA Department Distribution:")
    print(eda_stats["department_distribution"])
    print("\nEDA Sample Numerical Stats:")
    print(eda_stats["numerical_statistics"].head())
