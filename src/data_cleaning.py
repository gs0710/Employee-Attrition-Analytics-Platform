"""
data_cleaning.py - Robust data preparation, validation, anomaly correction, and audit tracking.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    PROCESSED_DATA_FILE,
    ID_COLUMN,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
)


def run_data_cleaning_pipeline(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes a comprehensive data cleaning and validation pipeline.
    Maintains complete audit logging of missing values, duplicates, and logical corrections.
    """
    raw_df = df.copy()
    audit_log = {
        "initial_rows": len(raw_df),
        "initial_cols": len(raw_df.columns),
        "missing_values_detected": {},
        "duplicates_removed": 0,
        "logical_corrections": {},
        "data_types_enforced": {},
        "cleaning_status": "Success",
    }

    # 1. Duplicate Record Detection
    initial_count = len(df)
    # Check duplicates by ID if present, else across all columns
    if ID_COLUMN in df.columns:
        df = df.drop_duplicates(subset=[ID_COLUMN], keep="first")
    else:
        df = df.drop_duplicates(keep="first")
    duplicates_count = initial_count - len(df)
    audit_log["duplicates_removed"] = int(duplicates_count)

    # If ID column is missing, generate unique identifiers
    if ID_COLUMN not in df.columns:
        df[ID_COLUMN] = [f"EMP-{i+1:04d}" for i in range(len(df))]

    # 2. Missing Value Detection and Documented Imputation
    missing_summary = df.isnull().sum()
    for col, count in missing_summary.items():
        if count > 0:
            audit_log["missing_values_detected"][col] = int(count)

    # Numerical imputation via Median
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            # Force numeric conversion in case of string numbers
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                audit_log.setdefault("imputations", {})[col] = {
                    "strategy": "median",
                    "value": float(median_val),
                }

    # Categorical imputation via Mode
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            if df[col].isnull().any():
                mode_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                df[col] = df[col].fillna(mode_val)
                audit_log.setdefault("imputations", {})[col] = {
                    "strategy": "mode",
                    "value": str(mode_val),
                }

    # 3. Logical Integrity Validation & Impossible Value Correction
    corrections = {}

    # Check 1: TotalWorkingYears >= YearsAtCompany
    if "TotalWorkingYears" in df.columns and "YearsAtCompany" in df.columns:
        violation_mask = df["TotalWorkingYears"] < df["YearsAtCompany"]
        if violation_mask.any():
            count = int(violation_mask.sum())
            corrections["TotalWorkingYears < YearsAtCompany"] = count
            # Adjust TotalWorkingYears to match YearsAtCompany
            df.loc[violation_mask, "TotalWorkingYears"] = df.loc[violation_mask, "YearsAtCompany"]

    # Check 2: YearsInCurrentRole <= YearsAtCompany
    if "YearsInCurrentRole" in df.columns and "YearsAtCompany" in df.columns:
        violation_mask = df["YearsInCurrentRole"] > df["YearsAtCompany"]
        if violation_mask.any():
            count = int(violation_mask.sum())
            corrections["YearsInCurrentRole > YearsAtCompany"] = count
            df.loc[violation_mask, "YearsInCurrentRole"] = df.loc[violation_mask, "YearsAtCompany"]

    # Check 3: YearsSinceLastPromotion <= YearsAtCompany
    if "YearsSinceLastPromotion" in df.columns and "YearsAtCompany" in df.columns:
        violation_mask = df["YearsSinceLastPromotion"] > df["YearsAtCompany"]
        if violation_mask.any():
            count = int(violation_mask.sum())
            corrections["YearsSinceLastPromotion > YearsAtCompany"] = count
            df.loc[violation_mask, "YearsSinceLastPromotion"] = df.loc[violation_mask, "YearsAtCompany"]

    # Check 4: YearsWithCurrManager <= YearsAtCompany
    if "YearsWithCurrManager" in df.columns and "YearsAtCompany" in df.columns:
        violation_mask = df["YearsWithCurrManager"] > df["YearsAtCompany"]
        if violation_mask.any():
            count = int(violation_mask.sum())
            corrections["YearsWithCurrManager > YearsAtCompany"] = count
            df.loc[violation_mask, "YearsWithCurrManager"] = df.loc[violation_mask, "YearsAtCompany"]

    # Check 5: Non-negative numerical values
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            negative_mask = df[col] < 0
            if negative_mask.any():
                count = int(negative_mask.sum())
                corrections[f"{col} < 0"] = count
                df.loc[negative_mask, col] = 0

    audit_log["logical_corrections"] = corrections
    audit_log["final_rows"] = len(df)
    audit_log["final_cols"] = len(df.columns)

    # Save cleaned dataset
    df.to_csv(PROCESSED_DATA_FILE, index=False)
    return df, audit_log


if __name__ == "__main__":
    from src.data_loader import load_or_generate_dataset
    raw = load_or_generate_dataset()
    cleaned_df, log = run_data_cleaning_pipeline(raw)
    print(f"Cleaned dataset rows: {len(cleaned_df)}")
    print(f"Cleaning audit log: {log}")
