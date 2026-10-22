"""
settings.py - Central configuration and paths for Employee Workforce Segmentation project.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHARTS_DIR = OUTPUTS_DIR / "charts"
TABLES_DIR = OUTPUTS_DIR / "tables"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Ensure all directories exist
for directory in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SAMPLE_DATA_DIR,
    MODELS_DIR,
    CHARTS_DIR,
    TABLES_DIR,
    REPORTS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# Default File Paths
RAW_DATA_FILE = RAW_DATA_DIR / "employee_data.csv"
SAMPLE_DATA_FILE = SAMPLE_DATA_DIR / "employee_sample.csv"
DATA_DICTIONARY_FILE = DATA_DIR / "data_dictionary.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "employee_processed.csv"
SEGMENTED_DATA_FILE = PROCESSED_DATA_DIR / "employee_segmented.csv"

KMEANS_MODEL_FILE = MODELS_DIR / "kmeans_model.joblib"
PREPROCESSOR_FILE = MODELS_DIR / "preprocessor.joblib"
PCA_MODEL_FILE = MODELS_DIR / "pca_model.joblib"

MAIN_REPORT_PDF = REPORTS_DIR / "employee_segmentation_report.pdf"
PCA_SUMMARY_PDF = REPORTS_DIR / "pca_summary.pdf"

# Reproducibility & Model Hyperparameters
RANDOM_STATE = 42
DEFAULT_K_CLUSTERS = 4
K_SEARCH_RANGE = (2, 8)
DENDROGRAM_MAX_SAMPLES = 60  # Readable dendrogram sample size

# Feature Definitions
ID_COLUMN = "EmployeeID"
ATTRITION_COLUMN = "Attrition"
DEPARTMENT_COLUMN = "Department"
JOB_ROLE_COLUMN = "JobRole"
PERFORMANCE_COLUMN = "PerformanceRating"

# Primary Numerical Features for Clustering
NUMERICAL_FEATURES = [
    "Age",
    "TotalWorkingYears",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
    "MonthlyIncome",
    "JobLevel",
    "TrainingTimesLastYear",
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "WorkLifeBalance",
    "AbsenteeismDays",
]

# Categorical Features for Clustering
CATEGORICAL_FEATURES = [
    "Department",
    "JobRole",
    "EducationLevel",
    "Overtime",
    "BusinessTravel",
]

# Sensitive/Excluded columns from primary clustering features
EXCLUDED_FEATURES = [
    ID_COLUMN,
    ATTRITION_COLUMN,  # Held out for evaluation & retention analysis, not clustering input
    "Gender",          # Excluded from distance metric to avoid bias
]

# Dashboard Styling & Theme
THEME_COLORS = {
    "primary": "#1E3A8A",      # Deep Slate Blue
    "secondary": "#0D9488",    # Teal
    "accent": "#F59E0B",       # Amber
    "danger": "#EF4444",       # Rose / Alert
    "dark": "#0F172A",         # Slate 900
    "light": "#F8FAFC",        # Slate 50
    "cluster_palette": ["#2563EB", "#059669", "#D97706", "#7C3AED", "#DB2777", "#0891B2"],
}

FAVICON_PATH = OUTPUTS_DIR / "favicon.png"

FLATICON_CDN_CSS = """
<link rel='stylesheet' href='https://cdn.jsdelivr.net/npm/@flaticon/flaticon-uicons@3.3.1/css/all/all.min.css'>
<style>
    /* Flaticon Uicons global styling */
    i.fi {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        vertical-align: -0.125em;
        line-height: 1;
    }
    .header-icon {
        font-size: 1.8rem;
        color: #1E3A8A;
        margin-right: 10px;
    }
    .section-icon {
        font-size: 1.3rem;
        color: #0D9488;
        margin-right: 8px;
    }
    .card-icon {
        font-size: 1.15rem;
        color: #2563EB;
        margin-right: 6px;
    }
    .badge-icon {
        font-size: 0.95rem;
        margin-right: 5px;
    }
    .nav-card-icon {
        font-size: 1.4rem;
        color: #1E3A8A;
        margin-right: 8px;
    }
</style>
"""

