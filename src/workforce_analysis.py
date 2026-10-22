"""
workforce_analysis.py - Department analytics, performance-satisfaction matrix, retention review indicators, and HR recommendations.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    TABLES_DIR,
    DEPARTMENT_COLUMN,
    ATTRITION_COLUMN,
    PERFORMANCE_COLUMN,
)


def analyze_department_segments(
    df: pd.DataFrame,
    cluster_col: str = "KMeans_Cluster",
    segment_names: Dict[int, str] = None,
) -> Dict[str, Any]:
    """
    Computes department-level workforce distributions, cluster composition,
    tenure, income, satisfaction, performance, and attrition statistics.
    """
    df_dept = df.copy()
    if segment_names and cluster_col in df_dept.columns:
        df_dept["Segment_Name"] = df_dept[cluster_col].map(segment_names)

    # 1. Department Cross-Tabulation Matrix
    dept_cluster_counts = pd.crosstab(
        df_dept[DEPARTMENT_COLUMN],
        df_dept[cluster_col],
    )
    dept_cluster_pcts = pd.crosstab(
        df_dept[DEPARTMENT_COLUMN],
        df_dept[cluster_col],
        normalize="index",
    ) * 100

    # 2. Department-level KPI Summary
    dept_summary_records = []
    for dept, sub in df_dept.groupby(DEPARTMENT_COLUMN):
        rec = {
            "Department": dept,
            "Headcount": len(sub),
            "Share_Pct": round((len(sub) / len(df_dept)) * 100, 2),
            "Avg_Tenure_Years": round(float(sub["YearsAtCompany"].mean()), 2) if "YearsAtCompany" in sub else np.nan,
            "Avg_Monthly_Income": round(float(sub["MonthlyIncome"].mean()), 2) if "MonthlyIncome" in sub else np.nan,
            "Avg_Job_Satisfaction": round(float(sub["JobSatisfaction"].mean()), 2) if "JobSatisfaction" in sub else np.nan,
            "Avg_Performance_Rating": round(float(sub[PERFORMANCE_COLUMN].mean()), 2) if PERFORMANCE_COLUMN in sub else np.nan,
        }
        if ATTRITION_COLUMN in sub:
            att_pct = (sub[ATTRITION_COLUMN].astype(str).str.lower() == "yes").mean() * 100
            rec["Observed_Attrition_Pct"] = round(float(att_pct), 2)
        if "Overtime" in sub:
            ot_pct = (sub["Overtime"].astype(str).str.lower() == "yes").mean() * 100
            rec["Overtime_Pct"] = round(float(ot_pct), 2)

        dept_summary_records.append(rec)

    dept_summary_df = pd.DataFrame(dept_summary_records)
    dept_summary_df.to_csv(TABLES_DIR / "department_segment_summary.csv", index=False)

    return {
        "contingency_counts": dept_cluster_counts,
        "contingency_percentages": dept_cluster_pcts.round(2),
        "department_summary": dept_summary_df,
    }


def analyze_performance_and_satisfaction(
    df: pd.DataFrame,
    cluster_col: str = "KMeans_Cluster",
    segment_names: Dict[int, str] = None,
) -> Dict[str, Any]:
    """
    Examines joint distribution of performance ratings and job satisfaction across clusters.
    Constructs a 4-quadrant strategic talent grid:
    - High Performance / High Satisfaction (Champions / Retain & Recognize)
    - High Performance / Low Satisfaction (Flight Risk / Urgent Engagement)
    - Low Performance / High Satisfaction (Coaching / Capability Building)
    - Low Performance / Low Satisfaction (Intensive Support & Restructuring)
    """
    df_perf = df.copy()
    if segment_names and cluster_col in df_perf.columns:
        df_perf["Segment_Name"] = df_perf[cluster_col].map(segment_names)

    # Performance distribution by cluster
    perf_col = PERFORMANCE_COLUMN if PERFORMANCE_COLUMN in df_perf.columns else "JobSatisfaction"
    sat_col = "JobSatisfaction" if "JobSatisfaction" in df_perf.columns else perf_col

    cluster_perf_summary = df_perf.groupby(cluster_col).agg(
        Headcount=("EmployeeID", "count"),
        Avg_Performance=(perf_col, "mean"),
        Avg_Satisfaction=(sat_col, "mean"),
        High_Performers=(perf_col, lambda s: (s >= 3).mean() * 100),
        Low_Performers=(perf_col, lambda s: (s < 2).mean() * 100),
    ).round(2).reset_index()

    if segment_names:
        cluster_perf_summary["Segment_Name"] = cluster_perf_summary[cluster_col].map(segment_names)

    # 4-Quadrant Categorization
    median_perf = df_perf[perf_col].median()
    median_sat = df_perf[sat_col].median()

    def categorize_quadrant(row):
        p = row[perf_col]
        s = row[sat_col]
        if p >= median_perf and s >= median_sat:
            return "Retain & Recognize (High Perf, High Sat)"
        elif p >= median_perf and s < median_sat:
            return "Urgent Review / Flight Risk (High Perf, Low Sat)"
        elif p < median_perf and s >= median_sat:
            return "Coaching & Enablement (Low Perf, High Sat)"
        else:
            return "Intensive Support Required (Low Perf, Low Sat)"

    df_perf["Talent_Quadrant"] = df_perf.apply(categorize_quadrant, axis=1)
    quadrant_counts = df_perf["Talent_Quadrant"].value_counts(normalize=True).mul(100).round(2).reset_index()
    quadrant_counts.columns = ["Talent_Quadrant", "Workforce_Pct"]

    cluster_perf_summary.to_csv(TABLES_DIR / "performance_summary.csv", index=False)

    return {
        "cluster_perf_summary": cluster_perf_summary,
        "quadrant_distribution": quadrant_counts,
        "df_with_quadrants": df_perf,
    }


def compute_retention_risk_indicators(
    df: pd.DataFrame,
    cluster_col: str = "KMeans_Cluster",
    segment_names: Dict[int, str] = None,
) -> pd.DataFrame:
    """
    Computes objective, descriptive retention review indicators per cluster.
    Indicators:
    1. Low Job Satisfaction (< 2.5)
    2. Low Work-Life Balance (< 2.5)
    3. High Overtime Prevalence (> 35%)
    4. Extended Promotion Stagnation (>= 4 years)
    5. High Absenteeism Days (> 6 days)
    6. Historical Attrition Rate vs Workforce Average (if Attrition field exists)
    """
    records = []
    total_wf = len(df)
    wf_attrition = (df[ATTRITION_COLUMN].astype(str).str.lower() == "yes").mean() * 100 if ATTRITION_COLUMN in df else np.nan

    for c_id, sub in df.groupby(cluster_col):
        n = len(sub)
        pct = (n / total_wf) * 100

        # Calculate specific flags
        low_sat_pct = (sub["JobSatisfaction"] <= 2).mean() * 100 if "JobSatisfaction" in sub else 0
        low_wlb_pct = (sub["WorkLifeBalance"] <= 2).mean() * 100 if "WorkLifeBalance" in sub else 0
        ot_pct = (sub["Overtime"].astype(str).str.lower() == "yes").mean() * 100 if "Overtime" in sub else 0
        stagnant_promo_pct = (sub["YearsSinceLastPromotion"] >= 4).mean() * 100 if "YearsSinceLastPromotion" in sub else 0
        high_absence_pct = (sub["AbsenteeismDays"] >= 6).mean() * 100 if "AbsenteeismDays" in sub else 0

        # Composite review score (0 to 100 index for prioritization)
        composite_risk_score = (
            (low_sat_pct * 0.25)
            + (low_wlb_pct * 0.20)
            + (ot_pct * 0.20)
            + (stagnant_promo_pct * 0.20)
            + (high_absence_pct * 0.15)
        )

        rec = {
            "Cluster_ID": int(c_id),
            "Segment_Name": segment_names.get(int(c_id), f"Cluster {c_id}") if segment_names else f"Cluster {c_id}",
            "Headcount": n,
            "Workforce_Share_Pct": round(pct, 2),
            "Composite_Review_Index": round(float(composite_risk_score), 2),
            "Low_Job_Satisfaction_Pct": round(float(low_sat_pct), 2),
            "Low_WorkLifeBalance_Pct": round(float(low_wlb_pct), 2),
            "Overtime_Prevalence_Pct": round(float(ot_pct), 2),
            "Promotion_Stagnant_Pct": round(float(stagnant_promo_pct), 2),
            "High_Absence_Pct": round(float(high_absence_pct), 2),
        }

        if not np.isnan(wf_attrition):
            c_att = (sub[ATTRITION_COLUMN].astype(str).str.lower() == "yes").mean() * 100
            rec["Observed_Attrition_Pct"] = round(float(c_att), 2)
            rec["Workforce_Attrition_Baseline_Pct"] = round(float(wf_attrition), 2)
            rec["Attrition_Delta"] = round(float(c_att - wf_attrition), 2)

        # Risk Classification (for supportive review)
        if composite_risk_score >= 38:
            rec["Review_Priority"] = "Elevated Review Recommended"
        elif composite_risk_score >= 25:
            rec["Review_Priority"] = "Moderate Attention"
        else:
            rec["Review_Priority"] = "Stable Engagement"

        records.append(rec)

    risk_df = pd.DataFrame(records).sort_values("Composite_Review_Index", ascending=False)
    risk_df.to_csv(TABLES_DIR / "retention_risk_analysis.csv", index=False)
    return risk_df


def generate_hr_recommendations(
    cluster_profiles_df: pd.DataFrame,
    risk_df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """
    Generates actionable, ethically sound, non-punitive workforce management recommendations
    aligned with the observed characteristics of each segment.
    """
    recommendations = []

    merged = cluster_profiles_df.merge(risk_df, on="Cluster_ID", suffixes=("", "_risk"))

    for _, row in merged.iterrows():
        c_id = int(row["Cluster_ID"])
        name = row["Segment_Name"]
        headcount = int(row["Headcount"])
        share = row["Workforce_Share_Pct"]
        priority = row["Review_Priority"]

        # Tailor recommendations based on segment profile attributes
        actions = []
        if "Senior Leadership" in name or row.get("Avg_MonthlyIncome", 0) > 10000:
            actions.extend([
                "Executive Knowledge Transfer: Implement structured succession planning and executive mentoring pipelines.",
                "Retention of Strategic IP: Establish long-term incentive grants and advisory tenure options.",
                "Leadership Bandwidth: Review administrative workload to maintain strategic focus.",
            ])
        elif "Early-Career" in name or row.get("Avg_Age", 40) < 32:
            actions.extend([
                "Accelerated Career Pathways: Provide defined 18-month rotational assignments and skill milestones.",
                "Mentorship Pairing: Pair emerging staff with senior mentors for career guidance and socialization.",
                "Learning & Development: Offer subsidized certification programs and high-visibility cross-functional projects.",
            ])
        elif "Stagnant" in name or row.get("Promotion_Stagnant_Pct", 0) > 30:
            actions.extend([
                "Promotion Path Transparency: Conduct career roadmap check-ins to clarify promotion criteria.",
                "Role Enrichment / Horizontal Mobility: Offer lateral role rotations to prevent skill stagnation.",
                "Workload & Overtime Audit: Rebalance team capacity where recurring overtime exceeds healthy thresholds.",
            ])
        else:
            actions.extend([
                "Core Employee Recognition: Maintain competitive compensation benchmarks and milestone awards.",
                "Work-Life Integration: Expand flexible scheduling and hybrid work accommodations.",
                "Continuous Feedback Loops: Leverage quarterly pulse surveys to sustain stable engagement.",
            ])

        recommendations.append({
            "Cluster_ID": c_id,
            "Segment_Name": name,
            "Headcount": headcount,
            "Workforce_Share_Pct": share,
            "Review_Priority": priority,
            "Actionable_Recommendations": actions,
            "Ethical_Guardrail": "Segment insights are intended solely for organizational resource planning, mentoring, and support. Individual employment, compensation, or termination actions based on cluster membership are strictly prohibited.",
        })

    return recommendations


if __name__ == "__main__":
    from src.data_loader import load_or_generate_dataset
    from src.data_cleaning import run_data_cleaning_pipeline
    from src.feature_engineering import WorkforceFeaturePipeline
    from src.clustering import train_kmeans_model, generate_cluster_profiles

    raw_df = load_or_generate_dataset()
    cleaned_df, _ = run_data_cleaning_pipeline(raw_df)

    feat_pipe = WorkforceFeaturePipeline()
    X_arr, _ = feat_pipe.fit_transform(cleaned_df)

    _, labels, _ = train_kmeans_model(X_arr, n_clusters=4)
    profiles_df, _, names = generate_cluster_profiles(cleaned_df, labels)
    cleaned_df["KMeans_Cluster"] = labels

    dept_res = analyze_department_segments(cleaned_df, segment_names=names)
    print("Department Summary:")
    print(dept_res["department_summary"])

    perf_res = analyze_performance_and_satisfaction(cleaned_df, segment_names=names)
    print("\nQuadrant Distribution:")
    print(perf_res["quadrant_distribution"])

    risk_df = compute_retention_risk_indicators(cleaned_df, segment_names=names)
    print("\nRetention Risk Indicators:")
    print(risk_df[["Cluster_ID", "Segment_Name", "Composite_Review_Index", "Review_Priority", "Observed_Attrition_Pct"]])

    recs = generate_hr_recommendations(profiles_df, risk_df)
    print(f"\nGenerated HR recommendations for {len(recs)} segments.")
