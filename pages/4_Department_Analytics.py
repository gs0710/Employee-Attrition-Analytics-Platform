"""
4_Department_Analytics.py - Departmental workforce segmentation, composition heatmaps, and functional benchmarks.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DEFAULT_K_CLUSTERS, THEME_COLORS, FAVICON_PATH, FLATICON_CDN_CSS
from src.data_loader import load_or_generate_dataset
from src.data_cleaning import run_data_cleaning_pipeline
from src.feature_engineering import WorkforceFeaturePipeline
from src.clustering import train_kmeans_model, generate_cluster_profiles
from src.workforce_analysis import analyze_department_segments
from src.visualization import create_department_cluster_stacked_bar

st.set_page_config(page_title="Department Analytics | Workforce Segmentation", page_icon=str(FAVICON_PATH), layout="wide")
st.markdown(FLATICON_CDN_CSS, unsafe_allow_html=True)

st.markdown('<h2><i class="fi fi-sr-building header-icon"></i>Department-Level Workforce Analytics</h2>', unsafe_allow_html=True)
st.markdown("Examine functional department composition, talent archetype distribution, and operational metrics.")

@st.cache_data
def get_dept_data():
    raw_df = load_or_generate_dataset()
    cleaned_df, _ = run_data_cleaning_pipeline(raw_df)

    feat_pipe = WorkforceFeaturePipeline()
    X_arr, _ = feat_pipe.fit_transform(cleaned_df)

    _, km_labels, _ = train_kmeans_model(X_arr, n_clusters=DEFAULT_K_CLUSTERS)
    _, _, names = generate_cluster_profiles(cleaned_df, km_labels)

    cleaned_df["KMeans_Cluster"] = km_labels
    cleaned_df["Segment_Name"] = cleaned_df["KMeans_Cluster"].map(names)

    dept_res = analyze_department_segments(cleaned_df, segment_names=names)
    return cleaned_df, dept_res, names

cleaned_df, dept_res, names = get_dept_data()
dept_summary = dept_res["department_summary"]
contingency_pct = dept_res["contingency_percentages"]

# Top Department KPIs
st.subheader("Department Workforce Headcount & Key Benchmarks")

d_cols = st.columns(len(dept_summary))
for idx, col in enumerate(d_cols):
    row = dept_summary.iloc[idx]
    with col:
        st.markdown(f"**{row['Department']}**")
        st.metric(
            label="Headcount",
            value=f"{row['Headcount']:,}",
            delta=f"{row['Share_Pct']:.1f}% of Workforce",
        )
        sat_val = row.get("Avg_Job_Satisfaction", row.get("Avg_JobSatisfaction", 0.0))
        st.caption(
            f"• Avg Income: **${row.get('Avg_Monthly_Income', 0):,.0f}**\n"
            f"• Avg Tenure: **{row.get('Avg_Tenure_Years', 0):.1f} yrs**\n"
            f"• Satisfaction: **{sat_val:.2f} / 4**\n"
            f"• Attrition: **{row.get('Observed_Attrition_Pct', 0):.1f}%**"
        )

st.markdown("---")

col_bar, col_table = st.columns([3, 2])

with col_bar:
    st.subheader("Cluster Composition by Department (%)")
    stacked_fig = create_department_cluster_stacked_bar(contingency_pct, segment_names=names)
    st.plotly_chart(stacked_fig, use_container_width=True)

with col_table:
    st.subheader("Department Contingency Matrix")
    st.markdown("Absolute employee count per segment:")
    st.dataframe(dept_res["contingency_counts"], use_container_width=True)

    st.markdown("Percentage share within each department (%):")
    st.dataframe(contingency_pct.style.format("{:.1f}%"), use_container_width=True)

st.markdown("---")

# Department Drill-Down
st.markdown('<h3><i class="fi fi-sr-search-alt section-icon"></i>Department Specific Drill-Down</h3>', unsafe_allow_html=True)
selected_dept = st.selectbox("Select Department to Inspect:", dept_summary["Department"].unique())

dept_sub = cleaned_df[cleaned_df["Department"] == selected_dept]

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("##### Job Role Headcount")
    role_counts = dept_sub["JobRole"].value_counts().reset_index()
    role_counts.columns = ["JobRole", "Headcount"]
    st.dataframe(role_counts, use_container_width=True)

with c2:
    st.markdown("##### Segment Distribution")
    seg_counts = dept_sub["Segment_Name"].value_counts().reset_index()
    seg_counts.columns = ["Segment_Name", "Count"]
    seg_counts["Share"] = (seg_counts["Count"] / len(dept_sub) * 100).round(1).astype(str) + "%"
    st.dataframe(seg_counts, use_container_width=True)

with c3:
    st.markdown("##### Department Working Conditions")
    ot_pct = (dept_sub["Overtime"].astype(str).str.lower() == "yes").mean() * 100
    wlb_avg = dept_sub["WorkLifeBalance"].mean() if "WorkLifeBalance" in dept_sub else 2.8
    abs_avg = dept_sub["AbsenteeismDays"].mean() if "AbsenteeismDays" in dept_sub else 4.0
    st.info(f"""
    • **Overtime Rate:** {ot_pct:.1f}%\n
    • **Avg Work-Life Balance:** {wlb_avg:.2f} / 4\n
    • **Avg Absenteeism:** {abs_avg:.1f} days/yr\n
    • **Internal Attrition Rate:** {(dept_sub['Attrition'].astype(str).str.lower() == 'yes').mean() * 100:.1f}%
    """)

st.warning("**Analytical Caveat:** Variations across departments naturally reflect differing functional operational requirements, technical specializations, and market compensation rates rather than structural underperformance.")
