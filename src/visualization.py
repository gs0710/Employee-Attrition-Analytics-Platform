"""
visualization.py - Interactive Plotly and publication-quality Matplotlib/Seaborn visualization generators.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless server execution
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import CHARTS_DIR, THEME_COLORS, DENDROGRAM_MAX_SAMPLES


# ==========================================
# 1. Clustering Evaluation Plots
# ==========================================

def create_clustering_evaluation_plot(eval_df: pd.DataFrame) -> go.Figure:
    """Creates a dual-axis interactive Plotly chart for Inertia (Elbow) and Silhouette Score."""
    fig = go.Figure()

    # Inertia line (Elbow)
    fig.add_trace(go.Scatter(
        x=eval_df["k"],
        y=eval_df["Inertia"],
        mode="lines+markers",
        name="Inertia (Elbow)",
        line=dict(color=THEME_COLORS["primary"], width=3),
        marker=dict(size=8),
        yaxis="y1",
    ))

    # Silhouette Score line
    fig.add_trace(go.Scatter(
        x=eval_df["k"],
        y=eval_df["Silhouette_Score"],
        mode="lines+markers",
        name="Silhouette Score",
        line=dict(color=THEME_COLORS["secondary"], width=3, dash="dot"),
        marker=dict(size=8),
        yaxis="y2",
    ))

    fig.update_layout(
        title="<b>K-Means Evaluation: Elbow Method vs. Silhouette Score</b>",
        xaxis=dict(title="Number of Clusters (k)", dtick=1),
        yaxis=dict(
            title=dict(text="Inertia (Within-Cluster Sum of Squares)", font=dict(color=THEME_COLORS["primary"])),
        ),
        yaxis2=dict(
            title=dict(text="Silhouette Score (Separation)", font=dict(color=THEME_COLORS["secondary"])),
            overlaying="y",
            side="right",
        ),
        legend=dict(x=0.05, y=0.95),
        template="plotly_white",
        height=450,
    )
    return fig


def save_static_kmeans_eval_plot(eval_df: pd.DataFrame, save_path: Path = CHARTS_DIR / "kmeans_eval.png") -> Path:
    """Scree/Elbow plot for ReportLab PDF report."""
    fig, ax1 = plt.subplots(figsize=(8, 4.5), dpi=300)

    color = THEME_COLORS["primary"]
    ax1.set_xlabel("Number of Clusters (k)", fontweight="bold")
    ax1.set_ylabel("Inertia (Within-Cluster SSE)", color=color, fontweight="bold")
    ax1.plot(eval_df["k"], eval_df["Inertia"], marker="o", color=color, linewidth=2)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    color2 = THEME_COLORS["secondary"]
    ax2.set_ylabel("Silhouette Score", color=color2, fontweight="bold")
    ax2.plot(eval_df["k"], eval_df["Silhouette_Score"], marker="s", color=color2, linewidth=2, linestyle="--")
    ax2.tick_params(axis="y", labelcolor=color2)

    plt.title("K-Means Model Evaluation Across Cluster Counts", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


# ==========================================
# 2. Hierarchical Dendrogram Plot
# ==========================================

def generate_dendrogram_plot(
    Z: np.ndarray,
    sample_size: int = DENDROGRAM_MAX_SAMPLES,
    save_path: Path = CHARTS_DIR / "hierarchical_dendrogram.png",
) -> Path:
    """Generates and saves a publication-quality dendrogram using SciPy."""
    plt.figure(figsize=(10, 5), dpi=300)
    dendrogram(
        Z,
        truncate_mode=None,
        leaf_rotation=90,
        leaf_font_size=8,
        color_threshold=0.7 * np.max(Z[:, 2]),
        above_threshold_color="grey",
    )
    plt.title(
        f"Hierarchical Clustering Dendrogram (Ward Linkage, Sample N={sample_size})",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    plt.xlabel("Sampled Employee Index / Merged Nodes", fontweight="bold")
    plt.ylabel("Ward Distance (Variance Threshold)", fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


# ==========================================
# 3. PCA Variance Charts
# ==========================================

def create_pca_variance_plot(variance_df: pd.DataFrame, max_components: int = 15) -> go.Figure:
    """Interactive scree plot and cumulative explained variance curve."""
    df_plot = variance_df.head(max_components)

    fig = go.Figure()

    # Individual explained variance bars
    fig.add_trace(go.Bar(
        x=df_plot["Component"],
        y=df_plot["Explained_Variance_Pct"],
        name="Individual Variance (%)",
        marker_color=THEME_COLORS["primary"],
        opacity=0.8,
    ))

    # Cumulative variance line
    fig.add_trace(go.Scatter(
        x=df_plot["Component"],
        y=df_plot["Cumulative_Variance_Pct"],
        name="Cumulative Variance (%)",
        mode="lines+markers",
        line=dict(color=THEME_COLORS["accent"], width=3),
        marker=dict(size=7),
    ))

    # Reference lines for 80% and 90%
    fig.add_hline(y=80, line_dash="dash", line_color="green", annotation_text="80% Threshold")
    fig.add_hline(y=90, line_dash="dash", line_color="orange", annotation_text="90% Threshold")

    fig.update_layout(
        title="<b>PCA Scree Plot & Cumulative Explained Variance</b>",
        xaxis_title="Principal Component",
        yaxis_title="Explained Variance Percentage (%)",
        yaxis=dict(range=[0, 105]),
        template="plotly_white",
        legend=dict(x=0.02, y=0.98),
        height=450,
    )
    return fig


def save_static_pca_variance_plot(variance_df: pd.DataFrame, save_path: Path = CHARTS_DIR / "pca_scree.png") -> Path:
    """Saves static PCA scree plot for reports."""
    df_plot = variance_df.head(12)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)

    bars = ax.bar(df_plot["Component"], df_plot["Explained_Variance_Pct"], color=THEME_COLORS["primary"], alpha=0.8, label="Individual %")
    ax.plot(df_plot["Component"], df_plot["Cumulative_Variance_Pct"], color=THEME_COLORS["accent"], marker="o", linewidth=2, label="Cumulative %")

    ax.axhline(80, color="green", linestyle="--", alpha=0.7, label="80% Threshold")
    ax.axhline(90, color="red", linestyle="--", alpha=0.7, label="90% Threshold")

    ax.set_ylabel("Explained Variance (%)", fontweight="bold")
    ax.set_xlabel("Principal Components", fontweight="bold")
    ax.set_title("PCA Scree Plot & Cumulative Variance", fontsize=12, fontweight="bold")
    ax.legend(loc="center right")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


# ==========================================
# 4. 2D PCA Cluster Scatter Plot
# ==========================================

def create_pca_2d_scatter_plot(
    df: pd.DataFrame,
    pca_coords: np.ndarray,
    color_col: str,
    title: str = "Workforce Clusters in 2D PCA Space",
    pc1_var: float = 26.5,
    pc2_var: float = 11.1,
) -> go.Figure:
    """
    Renders 2D PCA projection scatter plot with rich hover context
    (Anonymized Employee ID, Department, Income, Satisfaction, etc.).
    """
    plot_df = df.copy()
    plot_df["PC1"] = pca_coords[:, 0]
    plot_df["PC2"] = pca_coords[:, 1]
    plot_df[color_col] = plot_df[color_col].astype(str)

    hover_data = {
        "PC1": ":.2f",
        "PC2": ":.2f",
        "Department": True,
        "JobRole": True,
        "MonthlyIncome": ":$,.0f",
        "YearsAtCompany": True,
        "JobSatisfaction": True,
    }
    if "EmployeeID" in plot_df:
        hover_data["EmployeeID"] = True

    fig = px.scatter(
        plot_df,
        x="PC1",
        y="PC2",
        color=color_col,
        hover_name="EmployeeID" if "EmployeeID" in plot_df else None,
        hover_data=hover_data,
        color_discrete_sequence=THEME_COLORS["cluster_palette"],
        title=f"<b>{title}</b>",
        labels={
            "PC1": f"PC1: Career Longevity & Tenure ({pc1_var:.1f}% Variance)",
            "PC2": f"PC2: Hierarchical Grade & Compensation ({pc2_var:.1f}% Variance)",
            color_col: "Segment Group",
        },
        template="plotly_white",
        height=550,
    )
    fig.update_traces(marker=dict(size=7, opacity=0.85, line=dict(width=0.5, color="white")))
    return fig


def save_static_pca_scatter(
    df: pd.DataFrame,
    pca_coords: np.ndarray,
    cluster_labels: np.ndarray,
    save_path: Path = CHARTS_DIR / "pca_clusters_2d.png",
) -> Path:
    """Saves static 2D PCA projection for PDF report."""
    plt.figure(figsize=(8, 5.5), dpi=300)
    sns.scatterplot(
        x=pca_coords[:, 0],
        y=pca_coords[:, 1],
        hue=cluster_labels,
        palette=THEME_COLORS["cluster_palette"][: len(np.unique(cluster_labels))],
        alpha=0.85,
        s=35,
    )
    plt.title("Employee Workforce Segments in 2D PCA Space", fontsize=12, fontweight="bold", pad=10)
    plt.xlabel("PC1 (Career Longevity & Tenure)", fontweight="bold")
    plt.ylabel("PC2 (Hierarchical Grade & Income)", fontweight="bold")
    plt.legend(title="Cluster ID", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


# ==========================================
# 5. Component Loadings Visualization
# ==========================================

def create_loadings_bar_chart(top_df: pd.DataFrame, component_name: str = "PC1") -> go.Figure:
    """Horizontal bar chart of leading positive and negative feature loadings."""
    sorted_df = top_df.sort_values("Loading", ascending=True)

    colors = [THEME_COLORS["secondary"] if x >= 0 else THEME_COLORS["danger"] for x in sorted_df["Loading"]]

    fig = go.Figure(go.Bar(
        x=sorted_df["Loading"],
        y=sorted_df["Feature"],
        orientation="h",
        marker_color=colors,
    ))

    fig.update_layout(
        title=f"<b>Leading Feature Loadings for {component_name}</b>",
        xaxis_title="Loading Coefficient (Correlation with Component)",
        yaxis_title="Feature Attribute",
        template="plotly_white",
        height=400,
    )
    return fig


# ==========================================
# 6. Correlation Heatmap
# ==========================================

def save_static_correlation_heatmap(corr_df: pd.DataFrame, save_path: Path = CHARTS_DIR / "correlation_matrix.png") -> Path:
    """Saves correlation heatmap for EDA and reports."""
    plt.figure(figsize=(10, 8), dpi=300)
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    sns.heatmap(
        corr_df,
        mask=mask,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 7},
    )
    plt.title("Workforce Numerical Feature Correlation Heatmap", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


# ==========================================
# 7. Department & Cluster Composition
# ==========================================

def create_department_cluster_stacked_bar(crosstab_pct: pd.DataFrame, segment_names: Dict[int, str] = None) -> go.Figure:
    """Renders 100% stacked bar chart of cluster composition per department."""
    plot_df = crosstab_pct.copy()
    if segment_names:
        plot_df.columns = [segment_names.get(int(c), f"Cluster {c}") for c in plot_df.columns]

    fig = go.Figure()
    for col in plot_df.columns:
        fig.add_trace(go.Bar(
            name=str(col),
            x=plot_df.index,
            y=plot_df[col],
        ))

    fig.update_layout(
        barmode="stack",
        title="<b>Department-Level Segment Composition (%)</b>",
        xaxis_title="Department",
        yaxis_title="Workforce Percentage (%)",
        template="plotly_white",
        legend_title="Workforce Segment",
        height=450,
    )
    return fig


if __name__ == "__main__":
    from src.data_loader import load_or_generate_dataset
    from src.data_cleaning import run_data_cleaning_pipeline
    from src.feature_engineering import WorkforceFeaturePipeline
    from src.clustering import evaluate_kmeans_candidates, train_kmeans_model
    from src.pca_analysis import WorkforcePCAModel
    from src.hierarchical_clustering import compute_scipy_linkage

    raw_df = load_or_generate_dataset()
    cleaned_df, _ = run_data_cleaning_pipeline(raw_df)

    feat_pipe = WorkforceFeaturePipeline()
    X_arr, _ = feat_pipe.fit_transform(cleaned_df)

    eval_df = evaluate_kmeans_candidates(X_arr, k_range=(2, 6))
    save_static_kmeans_eval_plot(eval_df)

    Z, _ = compute_scipy_linkage(X_arr, sample_size=50)
    generate_dendrogram_plot(Z, sample_size=50)

    pca_model = WorkforcePCAModel()
    X_pca, pca_res = pca_model.fit_transform(X_arr, feat_pipe.transformed_feature_names)
    save_static_pca_variance_plot(pca_res["variance_table"])

    _, km_labels, _ = train_kmeans_model(X_arr, n_clusters=4)
    save_static_pca_scatter(cleaned_df, X_pca, km_labels)

    corr = cleaned_df.select_dtypes(include=[np.number]).corr().round(2)
    save_static_correlation_heatmap(corr)

    print("Successfully generated all static publication charts in outputs/charts/!")
