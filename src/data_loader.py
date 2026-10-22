"""
data_loader.py - Synthetic HR dataset generator, schema validation, and dataset loading utilities.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path if not present
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    RAW_DATA_FILE,
    SAMPLE_DATA_FILE,
    DATA_DICTIONARY_FILE,
    RANDOM_STATE,
    ID_COLUMN,
)


def generate_synthetic_workforce_data(n_samples: int = 1470, random_seed: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Generates a reproducible, structurally realistic synthetic workforce dataset
    modeled on standard HR analytics benchmarks (e.g., IBM HR analytics schema).
    Ensures logical consistency (e.g., TotalWorkingYears >= YearsAtCompany).
    """
    rng = np.random.default_rng(random_seed)

    # 1. Unique Identifiers
    employee_ids = [f"EMP-{i+1:04d}" for i in range(n_samples)]

    # 2. Departments & Associated Roles
    dept_choices = ["Research & Development", "Sales", "Human Resources"]
    dept_weights = [0.65, 0.28, 0.07]
    departments = rng.choice(dept_choices, size=n_samples, p=dept_weights)

    roles_by_dept = {
        "Research & Development": [
            "Research Scientist",
            "Software Engineer",
            "Laboratory Technician",
            "Manufacturing Director",
            "Research Director",
        ],
        "Sales": ["Sales Executive", "Sales Representative", "Manager"],
        "Human Resources": ["Human Resources", "Manager"],
    }
    role_weights = {
        "Research & Development": [0.30, 0.30, 0.22, 0.10, 0.08],
        "Sales": [0.60, 0.28, 0.12],
        "Human Resources": [0.75, 0.25],
    }

    job_roles = []
    for dept in departments:
        roles = roles_by_dept[dept]
        weights = role_weights[dept]
        job_roles.append(rng.choice(roles, p=weights))

    # 3. Demographics
    ages = rng.integers(low=20, high=60, size=n_samples)
    genders = rng.choice(["Male", "Female", "Non-Binary"], size=n_samples, p=[0.55, 0.42, 0.03])
    education_levels = rng.choice(
        ["Below College", "College", "Bachelor", "Master", "Doctor"],
        size=n_samples,
        p=[0.12, 0.20, 0.40, 0.23, 0.05],
    )

    # 4. Tenure & Experience Logic (Ensuring internal consistency)
    total_working_years = []
    years_at_company = []
    years_in_curr_role = []
    years_since_promo = []
    years_with_mgr = []
    job_levels = []
    monthly_incomes = []

    for i in range(n_samples):
        age = ages[i]
        role = job_roles[i]

        # Working years bound by age (started working around age 18-24)
        max_possible_work = max(0, age - 20)
        if max_possible_work <= 0:
            work_years = 0
        else:
            mode_val = min(max_possible_work, 10)
            work_years = int(rng.triangular(left=0, mode=mode_val, right=max_possible_work))
        total_working_years.append(work_years)

        # Company tenure <= Total working years
        max_tenure = min(work_years, max(0, age - 20))
        if max_tenure <= 0:
            tenure = 0
        else:
            mode_tenure = min(max_tenure, 5)
            tenure = int(rng.triangular(left=0, mode=mode_tenure, right=max_tenure))
        years_at_company.append(tenure)

        # Role, promotion, manager tenures <= Company tenure
        role_tenure = int(rng.integers(low=0, high=tenure + 1)) if tenure > 0 else 0
        promo_tenure = int(rng.integers(low=0, high=tenure + 1)) if tenure > 0 else 0
        mgr_tenure = int(rng.integers(low=0, high=tenure + 1)) if tenure > 0 else 0

        years_in_curr_role.append(role_tenure)
        years_since_promo.append(promo_tenure)
        years_with_mgr.append(mgr_tenure)

        # Job Level correlated with role & experience
        if "Director" in role or role == "Manager":
            lvl = rng.choice([4, 5], p=[0.6, 0.4])
        elif "Senior" in role or "Executive" in role or work_years > 10:
            lvl = rng.choice([2, 3, 4], p=[0.3, 0.5, 0.2])
        elif role in ["Sales Representative", "Laboratory Technician"] or work_years < 3:
            lvl = 1
        else:
            lvl = rng.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
        job_levels.append(lvl)

        # Base Monthly Income by Job Level with natural variation
        level_base_salary = {1: 2800, 2: 5200, 3: 8500, 4: 13500, 5: 18500}
        base_salary = level_base_salary[lvl]
        salary_noise = rng.normal(loc=0, scale=level_base_salary[lvl] * 0.12)
        income = max(1800, int(base_salary + salary_noise + (work_years * 120)))
        monthly_incomes.append(income)

    # 5. Satisfaction & Ratings (Likert 1 to 4)
    job_satisfaction = rng.choice([1, 2, 3, 4], size=n_samples, p=[0.16, 0.20, 0.35, 0.29])
    env_satisfaction = rng.choice([1, 2, 3, 4], size=n_samples, p=[0.15, 0.22, 0.34, 0.29])
    work_life_balance = rng.choice([1, 2, 3, 4], size=n_samples, p=[0.10, 0.24, 0.46, 0.20])
    performance_rating = rng.choice([1, 2, 3, 4], size=n_samples, p=[0.05, 0.15, 0.65, 0.15])

    training_times = rng.choice([0, 1, 2, 3, 4, 5, 6], size=n_samples, p=[0.04, 0.08, 0.36, 0.32, 0.12, 0.06, 0.02])
    absenteeism_days = rng.poisson(lam=4.2, size=n_samples)
    absenteeism_days = np.clip(absenteeism_days, 0, 30)

    # 6. Work Environment & Work Style
    overtime = rng.choice(["Yes", "No"], size=n_samples, p=[0.28, 0.72])
    travel = rng.choice(["Non-Travel", "Travel_Rarely", "Travel_Frequently"], size=n_samples, p=[0.15, 0.70, 0.15])

    # 7. Observed Historical Attrition (realistic probability based on multiple risk factors)
    # Note: Used strictly for validation and descriptive retention analytics, NOT input to clustering!
    attrition = []
    for i in range(n_samples):
        prob = 0.10  # Baseline attrition rate
        if overtime[i] == "Yes":
            prob += 0.12
        if job_satisfaction[i] == 1:
            prob += 0.12
        elif job_satisfaction[i] == 4:
            prob -= 0.05
        if work_life_balance[i] == 1:
            prob += 0.08
        if years_since_promo[i] >= 6:
            prob += 0.07
        if monthly_incomes[i] < 3500:
            prob += 0.06
        if total_working_years[i] <= 2:
            prob += 0.08
        if travel[i] == "Travel_Frequently":
            prob += 0.05

        prob = np.clip(prob, 0.02, 0.85)
        attrition.append("Yes" if rng.random() < prob else "No")

    df = pd.DataFrame({
        "EmployeeID": employee_ids,
        "Department": departments,
        "JobRole": job_roles,
        "Age": ages,
        "Gender": genders,
        "EducationLevel": education_levels,
        "JobLevel": job_levels,
        "MonthlyIncome": monthly_incomes,
        "TotalWorkingYears": total_working_years,
        "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": years_in_curr_role,
        "YearsSinceLastPromotion": years_since_promo,
        "YearsWithCurrManager": years_with_mgr,
        "TrainingTimesLastYear": training_times,
        "JobSatisfaction": job_satisfaction,
        "EnvironmentSatisfaction": env_satisfaction,
        "WorkLifeBalance": work_life_balance,
        "PerformanceRating": performance_rating,
        "Overtime": overtime,
        "BusinessTravel": travel,
        "AbsenteeismDays": absenteeism_days,
        "Attrition": attrition,
    })

    return df


def create_and_save_data_dictionary() -> pd.DataFrame:
    """
    Creates the formal data dictionary defining each field, data type, role, and description.
    """
    data_dict = [
        {"Column": "EmployeeID", "DataType": "String", "Role": "Identifier", "Description": "Unique employee identification number (excluded from clustering)", "Range_Values": "EMP-0001 to EMP-1470"},
        {"Column": "Department", "DataType": "Categorical", "Role": "Context / Segmentation", "Description": "Functional department of the employee", "Range_Values": "R&D, Sales, HR"},
        {"Column": "JobRole", "DataType": "Categorical", "Role": "Clustering Feature", "Description": "Specific job title and operational function", "Range_Values": "9 unique job roles"},
        {"Column": "Age", "DataType": "Integer", "Role": "Clustering Feature", "Description": "Employee chronological age in years", "Range_Values": "20 to 60"},
        {"Column": "Gender", "DataType": "Categorical", "Role": "Demographic (Excluded)", "Description": "Reported gender identity (excluded to uphold fairness)", "Range_Values": "Male, Female, Non-Binary"},
        {"Column": "EducationLevel", "DataType": "Categorical", "Role": "Clustering Feature", "Description": "Highest completed educational attainment", "Range_Values": "Below College to Doctor"},
        {"Column": "JobLevel", "DataType": "Integer", "Role": "Clustering Feature", "Description": "Corporate hierarchical grade / seniority level", "Range_Values": "1 (Entry) to 5 (Executive)"},
        {"Column": "MonthlyIncome", "DataType": "Integer", "Role": "Clustering Feature", "Description": "Gross monthly compensation in USD", "Range_Values": "$1,800 to $25,000+"},
        {"Column": "TotalWorkingYears", "DataType": "Integer", "Role": "Clustering Feature", "Description": "Cumulative career experience in years across all employers", "Range_Values": "0 to 40"},
        {"Column": "YearsAtCompany", "DataType": "Integer", "Role": "Clustering Feature", "Description": "Tenure at current enterprise in years", "Range_Values": "0 to 35"},
        {"Column": "YearsInCurrentRole", "DataType": "Integer", "Role": "Clustering Feature", "Description": "Duration in current job assignment in years", "Range_Values": "0 to 20"},
        {"Column": "YearsSinceLastPromotion", "DataType": "Integer", "Role": "Clustering Feature", "Description": "Years elapsed since most recent promotion", "Range_Values": "0 to 18"},
        {"Column": "YearsWithCurrManager", "DataType": "Integer", "Role": "Clustering Feature", "Description": "Tenure under current supervisory manager in years", "Range_Values": "0 to 18"},
        {"Column": "TrainingTimesLastYear", "DataType": "Integer", "Role": "Clustering Feature", "Description": "Number of professional training events attended last year", "Range_Values": "0 to 6"},
        {"Column": "JobSatisfaction", "DataType": "Integer (Likert)", "Role": "Clustering Feature", "Description": "Self-reported satisfaction with job role (1=Low, 4=Very High)", "Range_Values": "1 to 4"},
        {"Column": "EnvironmentSatisfaction", "DataType": "Integer (Likert)", "Role": "Clustering Feature", "Description": "Self-reported satisfaction with work environment", "Range_Values": "1 to 4"},
        {"Column": "WorkLifeBalance", "DataType": "Integer (Likert)", "Role": "Clustering Feature", "Description": "Self-reported balance between work and personal life", "Range_Values": "1 to 4"},
        {"Column": "PerformanceRating", "DataType": "Integer (Likert)", "Role": "Evaluation / Grouping", "Description": "Annual performance evaluation score (1=Unsatisfactory, 4=Outstanding)", "Range_Values": "1 to 4"},
        {"Column": "Overtime", "DataType": "Categorical", "Role": "Clustering Feature", "Description": "Regular requirement to perform overtime duties", "Range_Values": "Yes, No"},
        {"Column": "BusinessTravel", "DataType": "Categorical", "Role": "Clustering Feature", "Description": "Frequency of work-related travel", "Range_Values": "Non-Travel, Travel_Rarely, Travel_Frequently"},
        {"Column": "AbsenteeismDays", "DataType": "Integer", "Role": "Clustering Feature", "Description": "Total recorded days of work absence over the past 12 months", "Range_Values": "0 to 30"},
        {"Column": "Attrition", "DataType": "Categorical", "Role": "Evaluation Only (Target Held Out)", "Description": "Observed historical employee separation status (excluded from clustering)", "Range_Values": "Yes, No"},
    ]
    df_dict = pd.DataFrame(data_dict)
    df_dict.to_csv(DATA_DICTIONARY_FILE, index=False)
    return df_dict


def load_or_generate_dataset(filepath: Path = RAW_DATA_FILE, force_generate: bool = False) -> pd.DataFrame:
    """
    Loads the workforce dataset from disk if present, or generates and caches it.
    """
    if filepath.exists() and not force_generate:
        df = pd.read_csv(filepath)
    else:
        df = generate_synthetic_workforce_data(n_samples=1470, random_seed=RANDOM_STATE)
        df.to_csv(filepath, index=False)
        # Also save small sample
        sample_df = df.head(50)
        sample_df.to_csv(SAMPLE_DATA_FILE, index=False)
        create_and_save_data_dictionary()
    return df


def validate_dataset_schema(df: pd.DataFrame) -> dict:
    """
    Validates uploaded or provided dataframe for workforce segmentation requirements.
    Returns status dict with detected columns, missing expected columns, and validity flag.
    """
    expected_cols = {
        "EmployeeID": False,  # optional, generated if missing
        "Department": True,
        "Age": True,
        "MonthlyIncome": True,
        "JobSatisfaction": False,
        "PerformanceRating": False,
        "Attrition": False,
    }
    missing_critical = []
    for col, is_critical in expected_cols.items():
        if is_critical and col not in df.columns:
            missing_critical.append(col)

    is_valid = len(missing_critical) == 0
    return {
        "is_valid": is_valid,
        "missing_critical": missing_critical,
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "has_attrition": "Attrition" in df.columns,
        "has_performance": "PerformanceRating" in df.columns,
    }


if __name__ == "__main__":
    df = load_or_generate_dataset(force_generate=True)
    print(f"Generated workforce dataset with shape: {df.shape}")
    print(f"Sample columns: {list(df.columns)}")
    print(f"Observed Attrition rate: {(df['Attrition'] == 'Yes').mean():.2%}")
