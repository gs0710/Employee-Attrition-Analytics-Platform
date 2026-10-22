"""
1_Data_Explorer.py - Interactive data exploration, schema validation, data dictionary, and file upload.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DATA_DICTIONARY_FILE, THEME_COLORS, FAVICON_PATH, FLATICON_CDN_CSS
from src.data_loader import load_or_generate_dataset, validate_dataset_schema
from src.data_cleaning import run_data_cleaning_pipeline
from src.eda import generate_eda_summary_statistics

st.set_page_config(page_title="Data Explorer | Workforce Segmentation", page_icon=str(FAVICON_PATH), layout="wide")
st.markdown(FLATICON_CDN_CSS, unsafe_allow_html=True)

st.markdown('<h2><i class="fi fi-sr-database header-icon"></i>Workforce Data Explorer & Schema Inspector</h2>', unsafe_allow_html=True)
st.markdown("Inspect raw and cleaned employee records, examine the data dictionary, and upload custom datasets.")

# Tabbed Interface
tab1, tab2, tab3, tab4 = st.tabs([
    "Dataset Overview & Records",
    "Data Dictionary",
    "Data Quality & Audit",
    "Custom CSV Upload",
])

# Default dataset
df = load_or_generate_dataset()
cleaned_df, audit_log = run_data_cleaning_pipeline(df)

with tab1:
    st.markdown('<h4><i class="fi fi-sr-table card-icon"></i>Enterprise Workforce Dataset</h4>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Total Features", len(df.columns))
    col3.metric("Numerical Attributes", len(df.select_dtypes(include=[np.number]).columns))
    col4.metric("Categorical Attributes", len(df.select_dtypes(include=['object']).columns))

    # Search & Column Filter
    st.markdown("##### Filter & Browse Records")
    col_search, col_dept = st.columns([2, 1])
    search_query = col_search.text_input("Search by Employee ID or Role:", "")
    dept_filter = col_dept.selectbox("Filter Department:", ["All"] + list(df["Department"].unique()))

    filtered_df = df.copy()
    if search_query:
        mask = (
            filtered_df["EmployeeID"].astype(str).str.contains(search_query, case=False, na=False) |
            filtered_df["JobRole"].astype(str).str.contains(search_query, case=False, na=False)
        )
        filtered_df = filtered_df[mask]

    if dept_filter != "All":
        filtered_df = filtered_df[filtered_df["Department"] == dept_filter]

    st.dataframe(filtered_df, use_container_width=True, height=400)
    st.caption(f"Showing {len(filtered_df):,} of {len(df):,} employee records.")

    # Numerical Summary Statistics
    with st.expander("View Descriptive Numerical Summary Statistics", expanded=False):
        num_stats = df.select_dtypes(include=[np.number]).describe().T
        st.dataframe(num_stats.style.format("{:.2f}"), use_container_width=True)

with tab2:
    st.markdown('<h4><i class="fi fi-sr-book-alt card-icon"></i>Workforce Attribute Data Dictionary</h4>', unsafe_allow_html=True)
    st.markdown("Formal field definitions, data types, semantic roles, and allowed values.")

    if DATA_DICTIONARY_FILE.exists():
        dict_df = pd.read_csv(DATA_DICTIONARY_FILE)
        st.dataframe(dict_df, use_container_width=True, height=450)
    else:
        st.info("Generating data dictionary...")
        from src.data_loader import create_and_save_data_dictionary
        dict_df = create_and_save_data_dictionary()
        st.dataframe(dict_df, use_container_width=True)

    csv_data = dict_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Data Dictionary (CSV)", data=csv_data, file_name="workforce_data_dictionary.csv", mime="text/csv")

with tab3:
    st.markdown('<h4><i class="fi fi-sr-shield-check card-icon"></i>Data Cleaning Audit & Integrity Report</h4>', unsafe_allow_html=True)
    st.markdown("Documented cleaning rules, missing value resolution, duplicate checks, and domain constraints.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Missing Values Detected", sum(audit_log.get("missing_values_detected", {}).values()))
    c2.metric("Duplicates Removed", audit_log.get("duplicates_removed", 0))
    c3.metric("Logical Rule Corrections", sum(audit_log.get("logical_corrections", {}).values()))

    st.markdown("##### Imputation & Correction Audit Log")
    st.json(audit_log)

    st.markdown("##### Domain Consistency Rules Enforced:")
    st.markdown("""
    1. **TotalWorkingYears >= YearsAtCompany:** Corrects anomalous entries where company tenure exceeds total lifetime career experience.
    2. **YearsInCurrentRole <= YearsAtCompany:** Ensures role duration does not exceed organizational tenure.
    3. **YearsSinceLastPromotion <= YearsAtCompany:** Ensures promotion tracking does not predate initial hiring.
    4. **Non-negative Bounds:** All continuous variables (tenure, income, absenteeism) are bounded strictly >= 0.
    5. **Categorical Imputation:** Categorical columns impute mode; continuous columns impute median to resist outlier skew.
    """)

with tab4:
    st.markdown('<h4><i class="fi fi-sr-cloud-download card-icon"></i>Custom Dataset Ingestion</h4>', unsafe_allow_html=True)
    st.markdown("Upload your organization's CSV file to evaluate clustering and segmentation on your own workforce.")

    uploaded_file = st.file_uploader("Upload Employee CSV File", type=["csv"])
    if uploaded_file is not None:
        try:
            user_df = pd.read_csv(uploaded_file)
            validation = validate_dataset_schema(user_df)

            if validation["is_valid"]:
                st.success(f"Dataset validated successfully! ({validation['row_count']} rows, {validation['column_count']} columns)")
                st.dataframe(user_df.head(10), use_container_width=True)
                st.info("This dataset meets all critical schema requirements and can be segmented using the models in the Clustering page.")
            else:
                st.error(f"Schema validation failed. Missing critical columns: {validation['missing_critical']}")
                st.warning("Please ensure your CSV includes: Department, Age, MonthlyIncome.")
        except Exception as e:
            st.error(f"Error parsing uploaded CSV: {e}")
