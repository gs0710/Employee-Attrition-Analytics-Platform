"""
5_Performance_Retention.py - Performance-satisfaction talent matrix, retention review indicators, and ethical HR recommendations.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DEFAULT_K_CLUSTERS, THEME_COLORS, FAVICON_PATH, FLATICON_CDN_CSS
from src.data_loader import load_or_generate_dataset
from src.data_cleaning import run_data_cleaning_pipeline
from src.feature_engineering import WorkforceFeaturePipeline
from src.clustering import train_kmeans_model, generate_cluster_profiles
from src.workforce_analysis import (
    analyze_performance_and_satisfaction,
    compute_retention_risk_indicators,
    generate_hr_recommendations,
)

st.set_page_config(page_title="Performance & Retention | Workforce Segmentation", page_icon=str(FAVICON_PATH), layout="wide")
st.markdown(FLATICON_CDN_CSS, unsafe_allow_html=True)

st.markdown('<h2><i class="fi fi-sr-bullseye-arrow header-icon"></i>Performance Groups & Retention Risk Diagnostics</h2>', unsafe_allow_html=True)
st.markdown("Examine the performance-satisfaction talent matrix, objective retention review indicators, and supportive HR initiatives.")

@st.cache_data
def get_perf_retention_data():
    raw_df = load_or_generate_dataset()
    cleaned_df, _ = run_data_cleaning_pipeline(raw_df)

    feat_pipe = WorkforceFeaturePipeline()
    X_arr, _ = feat_pipe.fit_transform(cleaned_df)

    _, km_labels, _ = train_kmeans_model(X_arr, n_clusters=DEFAULT_K_CLUSTERS)
    profiles_df, _, names = generate_cluster_profiles(cleaned_df, km_labels)

    cleaned_df["KMeans_Cluster"] = km_labels
    cleaned_df["Segment_Name"] = cleaned_df["KMeans_Cluster"].map(names)

    perf_res = analyze_performance_and_satisfaction(cleaned_df, segment_names=names)
    risk_df = compute_retention_risk_indicators(cleaned_df, segment_names=names)
    recs = generate_hr_recommendations(profiles_df, risk_df)

    return cleaned_df, profiles_df, perf_res, risk_df, recs

cleaned_df, profiles_df, perf_res, risk_df, recs = get_perf_retention_data()

tab_matrix, tab_risk, tab_recs = st.tabs([
    "Performance-Satisfaction Talent Grid",
    "Retention & Engagement Indicators",
    "Supportive HR Recommendations",
])

with tab_matrix:
    st.markdown('<h4><i class="fi fi-sr-apps card-icon"></i>Performance vs. Job Satisfaction Strategic Talent Matrix</h4>', unsafe_allow_html=True)
    st.markdown("""
    Aligning self-reported **Job Satisfaction** against **Annual Performance Rating** creates a strategic 4-quadrant talent grid
    to guide resource allocation, coaching, and proactive retention.
    """)

    col_q1, col_q2 = st.columns([1, 2])
    with col_q1:
        st.markdown("##### Workforce Quadrant Shares")
        q_df = perf_res["quadrant_distribution"]
        st.dataframe(q_df.style.format({"Workforce_Pct": "{:.1f}%"}), use_container_width=True)

    with col_q2:
        df_quad = perf_res["df_with_quadrants"]
        quad_fig = px.scatter(
            df_quad,
            x="JobSatisfaction",
            y="PerformanceRating",
            color="Talent_Quadrant",
            hover_name="EmployeeID",
            hover_data=["JobRole", "Department", "MonthlyIncome", "YearsAtCompany"],
            title="<b>Performance vs. Satisfaction Distribution</b>",
            labels={"JobSatisfaction": "Job Satisfaction (1-4)", "PerformanceRating": "Performance Rating (1-4)"},
            template="plotly_white",
            height=400,
        )
        quad_fig.update_traces(marker=dict(size=9, opacity=0.7))
        st.plotly_chart(quad_fig, use_container_width=True)

    st.markdown("##### Cluster-Level Performance Summary")
    st.dataframe(perf_res["cluster_perf_summary"].style.format({
        "Avg_Performance": "{:.2f}",
        "Avg_Satisfaction": "{:.2f}",
        "High_Performers": "{:.1f}%",
        "Low_Performers": "{:.1f}%",
    }), use_container_width=True)

with tab_risk:
    st.markdown('<h4><i class="fi fi-sr-shield-exclamation card-icon"></i>Retention & Engagement Review Indicators</h4>', unsafe_allow_html=True)
    st.markdown("""
    **Analytical Framework:** Rather than asserting individual employee flight predictions, this diagnostic tracks objective,
    job-related friction indicators: low satisfaction (<=2), low work-life balance (<=2), overtime frequency, and promotion stagnation (>=4 yrs).
    """)

    st.dataframe(risk_df.style.format({
        "Workforce_Share_Pct": "{:.1f}%",
        "Composite_Review_Index": "{:.1f}",
        "Low_Job_Satisfaction_Pct": "{:.1f}%",
        "Low_WorkLifeBalance_Pct": "{:.1f}%",
        "Overtime_Prevalence_Pct": "{:.1f}%",
        "Promotion_Stagnant_Pct": "{:.1f}%",
        "High_Absence_Pct": "{:.1f}%",
        "Observed_Attrition_Pct": "{:.1f}%",
        "Workforce_Attrition_Baseline_Pct": "{:.1f}%",
        "Attrition_Delta": "{:+.1f}%",
    }), use_container_width=True)

    st.info("""
    **Methodology Note on Retention Diagnostics:**
    - **Clustering is Descriptive, Not Predictive:** Unsupervised clustering segments the workforce by observational similarity. It does NOT generate individual flight probabilities.
    - **Baseline Context:** Observed historical attrition reflects previous separation patterns used strictly as an aggregate benchmark.
    - **Non-punitive Terminology:** Segments with higher friction indices are designated as *"Requires Further Review / Proactive Support"*, never labeled as 'bad' employees.
    """)

with tab_recs:
    st.markdown('<h4><i class="fi fi-sr-bulb card-icon"></i>Strategic & Supportive HR Recommendations</h4>', unsafe_allow_html=True)
    st.markdown("Actionable organizational interventions aligned with segment characteristics:")

    for r in recs:
        with st.expander(f"Cluster {r['Cluster_ID']}: {r['Segment_Name']} ({r['Headcount']} employees | {r['Workforce_Share_Pct']:.1f}%) — Priority: {r['Review_Priority']}", expanded=True):
            st.markdown(f"**Identified Archetype Priority:** `{r['Review_Priority']}`")
            st.markdown("##### Recommended Organizational Actions:")
            for action in r["Actionable_Recommendations"]:
                st.markdown(f"- {action}")
            st.markdown(f'<p style="font-size:0.85rem; color:#64748B;"><i class="fi fi-sr-shield-check badge-icon" style="color:#0D9488;"></i><strong>Ethical Guardrail:</strong> {r["Ethical_Guardrail"]}</p>', unsafe_allow_html=True)
