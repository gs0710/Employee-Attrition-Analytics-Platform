"""
test_data_cleaning.py - Unit tests for data loading, cleaning, validation, and domain rules.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import generate_synthetic_workforce_data, load_or_generate_dataset, validate_dataset_schema
from src.data_cleaning import run_data_cleaning_pipeline


def test_synthetic_data_generation():
    """Verifies synthetic dataset generation shape, reproducibility, and required fields."""
    df1 = generate_synthetic_workforce_data(n_samples=100, random_seed=42)
    df2 = generate_synthetic_workforce_data(n_samples=100, random_seed=42)

    assert len(df1) == 100
    assert "EmployeeID" in df1.columns
    assert "MonthlyIncome" in df1.columns
    assert "JobSatisfaction" in df1.columns
    # Verify strict reproducibility with fixed seed
    pd.testing.assert_frame_equal(df1, df2)


def test_missing_value_imputation():
    """Verifies that missing values in continuous and categorical features are correctly imputed."""
    df = generate_synthetic_workforce_data(n_samples=50, random_seed=42)

    # Inject missing values
    df.loc[0, "MonthlyIncome"] = np.nan
    df.loc[1, "Department"] = np.nan

    cleaned, audit = run_data_cleaning_pipeline(df)

    assert cleaned["MonthlyIncome"].isnull().sum() == 0
    assert cleaned["Department"].isnull().sum() == 0
    assert "MonthlyIncome" in audit.get("imputations", {})
    assert "Department" in audit.get("imputations", {})


def test_duplicate_record_handling():
    """Verifies that duplicate records are identified and purged."""
    df = generate_synthetic_workforce_data(n_samples=50, random_seed=42)

    # Duplicate row 0
    dup_row = df.iloc[[0]]
    df_with_dup = pd.concat([df, dup_row], ignore_index=True)
    assert len(df_with_dup) == 51

    cleaned, audit = run_data_cleaning_pipeline(df_with_dup)
    assert len(cleaned) == 50
    assert audit["duplicates_removed"] == 1


def test_logical_consistency_rules():
    """Verifies that impossible values (e.g. YearsAtCompany > TotalWorkingYears) are adjusted."""
    df = generate_synthetic_workforce_data(n_samples=20, random_seed=42)

    # Force impossible condition
    df.loc[0, "TotalWorkingYears"] = 2
    df.loc[0, "YearsAtCompany"] = 10
    df.loc[1, "YearsInCurrentRole"] = 15
    df.loc[1, "YearsAtCompany"] = 5

    cleaned, audit = run_data_cleaning_pipeline(df)

    # After cleaning, TotalWorkingYears should be adjusted to match YearsAtCompany
    assert cleaned.loc[0, "TotalWorkingYears"] >= cleaned.loc[0, "YearsAtCompany"]
    assert cleaned.loc[1, "YearsInCurrentRole"] <= cleaned.loc[1, "YearsAtCompany"]
    assert audit["logical_corrections"]["TotalWorkingYears < YearsAtCompany"] >= 1
    assert audit["logical_corrections"]["YearsInCurrentRole > YearsAtCompany"] >= 1


def test_schema_validation_utility():
    """Verifies that schema validation identifies missing critical columns."""
    valid_df = pd.DataFrame({
        "Department": ["Sales"],
        "Age": [30],
        "MonthlyIncome": [5000],
    })
    res = validate_dataset_schema(valid_df)
    assert res["is_valid"] is True
    assert len(res["missing_critical"]) == 0

    invalid_df = pd.DataFrame({"Age": [30]})
    res_inv = validate_dataset_schema(invalid_df)
    assert res_inv["is_valid"] is False
    assert "Department" in res_inv["missing_critical"]
