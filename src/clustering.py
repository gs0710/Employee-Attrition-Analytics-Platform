"""
clustering.py - K-Means clustering implementation, multi-metric evaluation, profiling, and naming.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    KMEANS_MODEL_FILE,
    RANDOM_STATE,
    DEFAULT_K_CLUSTERS,
    K_SEARCH_RANGE,
    ATTRITION_COLUMN,
)


def evaluate_kmeans_candidates(
    X: np.ndarray,
    k_range: Tuple[int, int] = K_SEARCH_RANGE,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Evaluates K-Means clustering across a range of k values using 4 standard metrics:
    - Inertia (Sum of squared distances, for Elbow method)
    - Silhouette Score (-1 to +1, higher is better)
    - Davies-Bouldin Index (>= 0, lower is better)
    - Calinski-Harabasz Index (higher is better)
    """
    results = []
    if isinstance(k_range, range):
        k_values = [k for k in k_range if 2 <= k < len(X)]
    elif isinstance(k_range, (list, tuple)) and len(k_range) == 2:
        min_k, max_k = int(k_range[0]), int(k_range[1])
        max_k = min(max_k, len(X) - 1)
        k_values = list(range(min_k, max_k + 1))
    elif isinstance(k_range, (list, tuple)):
        k_values = [int(k) for k in k_range if 2 <= k < len(X)]
    else:
        k_values = list(range(2, min(7, len(X))))

    for k in k_values:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state)
        labels = km.fit_predict(X)

        inertia = km.inertia_
        sil = silhouette_score(X, labels)
        db = davies_bouldin_score(X, labels)
        ch = calinski_harabasz_score(X, labels)

        results.append({
            "k": k,
            "Inertia": round(float(inertia), 2),
            "Silhouette_Score": round(float(sil), 4),
            "Davies_Bouldin_Index": round(float(db), 4),
            "Calinski_Harabasz_Index": round(float(ch), 2),
        })

    return pd.DataFrame(results)


def train_kmeans_model(
    X: np.ndarray,
    n_clusters: int = DEFAULT_K_CLUSTERS,
    random_state: int = RANDOM_STATE,
) -> Tuple[KMeans, np.ndarray, Dict[str, Any]]:
    """
    Fits K-Means estimator on processed feature matrix X.
    Returns the fitted model, predicted cluster labels, and performance evaluation metrics.
    """
    model = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=20,
        max_iter=300,
        random_state=random_state,
    )
    labels = model.fit_predict(X)

    # Compute clustering quality metrics
    sil = float(silhouette_score(X, labels))
    db = float(davies_bouldin_score(X, labels))
    ch = float(calinski_harabasz_score(X, labels))
    inertia = float(model.inertia_)

    metrics = {
        "n_clusters": n_clusters,
        "inertia": round(inertia, 2),
        "silhouette_score": round(sil, 4),
        "davies_bouldin_index": round(db, 4),
        "calinski_harabasz_index": round(ch, 2),
    }

    # Persist fitted model
    joblib.dump(model, KMEANS_MODEL_FILE)

    return model, labels, metrics


def generate_cluster_profiles(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    cluster_col_name: str = "KMeans_Cluster",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, str]]:
    """
    Computes comprehensive cluster profiles, workforce benchmarks, distinguishing features,
    and assigns descriptive, data-driven segment names based on actual cluster characteristics.
    """
    df_profile = df.copy()
    df_profile[cluster_col_name] = cluster_labels
    total_employees = len(df_profile)

    key_num_cols = [
        "Age",
        "MonthlyIncome",
        "YearsAtCompany",
        "TotalWorkingYears",
        "YearsSinceLastPromotion",
        "JobSatisfaction",
        "WorkLifeBalance",
        "PerformanceRating",
        "AbsenteeismDays",
    ]
    # Filter only available columns
    available_num = [c for c in key_num_cols if c in df_profile.columns]

    # Calculate overall workforce benchmark
    workforce_means = df_profile[available_num].mean()
    if ATTRITION_COLUMN in df_profile.columns:
        workforce_attrition = (df_profile[ATTRITION_COLUMN].astype(str).str.lower() == "yes").mean()
    else:
        workforce_attrition = np.nan

    cluster_records = []
    cluster_names = {}

    for c_id in sorted(np.unique(cluster_labels)):
        c_sub = df_profile[df_profile[cluster_col_name] == c_id]
        c_size = len(c_sub)
        c_pct = (c_size / total_employees) * 100

        rec = {
            "Cluster_ID": int(c_id),
            "Headcount": int(c_size),
            "Workforce_Share_Pct": round(c_pct, 2),
        }

        # Numeric averages
        for col in available_num:
            rec[f"Avg_{col}"] = round(float(c_sub[col].mean()), 2)

        # Historical Attrition Rate if available
        if ATTRITION_COLUMN in c_sub.columns:
            att_rate = (c_sub[ATTRITION_COLUMN].astype(str).str.lower() == "yes").mean() * 100
            rec["Observed_Attrition_Pct"] = round(float(att_rate), 2)

        # Overtime Rate
        if "Overtime" in c_sub.columns:
            ot_rate = (c_sub["Overtime"].astype(str).str.lower() == "yes").mean() * 100
            rec["Overtime_Rate_Pct"] = round(float(ot_rate), 2)

        # Derive intelligent data-driven name based on actual deviations from workforce means
        name = derive_cluster_name(c_sub, workforce_means, workforce_attrition)
        cluster_names[int(c_id)] = name
        rec["Segment_Name"] = name

        cluster_records.append(rec)

    profiles_df = pd.DataFrame(cluster_records)

    # Calculate distinguishing feature z-scores (comparison to workforce mean in std units)
    distinguishing_records = []
    workforce_stds = df_profile[available_num].std().replace(0, 1.0)

    for c_id in sorted(np.unique(cluster_labels)):
        c_sub = df_profile[df_profile[cluster_col_name] == c_id]
        for col in available_num:
            z_score = (c_sub[col].mean() - workforce_means[col]) / workforce_stds[col]
            distinguishing_records.append({
                "Cluster_ID": int(c_id),
                "Segment_Name": cluster_names[int(c_id)],
                "Feature": col,
                "Cluster_Mean": round(float(c_sub[col].mean()), 2),
                "Workforce_Mean": round(float(workforce_means[col]), 2),
                "Z_Score_Deviation": round(float(z_score), 2),
            })

    distinguishing_df = pd.DataFrame(distinguishing_records)

    return profiles_df, distinguishing_df, cluster_names


def derive_cluster_name(
    cluster_df: pd.DataFrame,
    workforce_means: pd.Series,
    workforce_attrition: float,
) -> str:
    """
    Dynamically infers an interpretable business segment name based on statistical anomalies.
    """
    avg_income = cluster_df["MonthlyIncome"].mean() if "MonthlyIncome" in cluster_df else 0
    avg_tenure = cluster_df["YearsAtCompany"].mean() if "YearsAtCompany" in cluster_df else 0
    avg_age = cluster_df["Age"].mean() if "Age" in cluster_df else 0
    avg_sat = cluster_df["JobSatisfaction"].mean() if "JobSatisfaction" in cluster_df else 2.5
    avg_promo = cluster_df["YearsSinceLastPromotion"].mean() if "YearsSinceLastPromotion" in cluster_df else 0

    wf_income = workforce_means.get("MonthlyIncome", 6000)
    wf_tenure = workforce_means.get("YearsAtCompany", 6)
    wf_age = workforce_means.get("Age", 37)
    wf_sat = workforce_means.get("JobSatisfaction", 2.7)

    # 1. Senior Executive / Veteran High-Earners
    if avg_income > wf_income * 1.35 and avg_tenure > wf_tenure * 1.2:
        return "Senior Leadership & Tenured Specialists"

    # 2. Early-Career / Emerging Talent
    elif avg_age < wf_age * 0.88 and avg_tenure < wf_tenure * 0.7:
        if avg_sat >= wf_sat:
            return "Early-Career High-Engagement Talent"
        else:
            return "Early-Career Onboarding Cohort"

    # 3. Promotion-Stagnant / Overextended Risk Group
    elif avg_promo > 4.5 or (avg_sat < wf_sat * 0.88):
        return "Stagnant & Retention-Review Segment"

    # 4. High-tenure Experienced Operational Backbone
    elif avg_tenure >= wf_tenure:
        return "Experienced Core Operational Backbone"

    # Fallback balanced descriptor
    else:
        return "Mid-Level Professional Contributors"


if __name__ == "__main__":
    from src.data_loader import load_or_generate_dataset
    from src.data_cleaning import run_data_cleaning_pipeline
    from src.feature_engineering import WorkforceFeaturePipeline

    raw_df = load_or_generate_dataset()
    cleaned_df, _ = run_data_cleaning_pipeline(raw_df)

    feat_pipe = WorkforceFeaturePipeline()
    X_arr, _ = feat_pipe.fit_transform(cleaned_df)

    # Candidate evaluation
    eval_df = evaluate_kmeans_candidates(X_arr, k_range=(2, 6))
    print("K-Means Candidate Evaluation:")
    print(eval_df)

    # Train model
    model, labels, metrics = train_kmeans_model(X_arr, n_clusters=4)
    print(f"\nFinal K-Means Metrics (k=4): {metrics}")

    # Generate profiles
    profiles_df, dist_df, names = generate_cluster_profiles(cleaned_df, labels)
    print("\nCluster Profiles Summary:")
    print(profiles_df[["Cluster_ID", "Segment_Name", "Headcount", "Workforce_Share_Pct", "Avg_MonthlyIncome", "Avg_YearsAtCompany"]])
