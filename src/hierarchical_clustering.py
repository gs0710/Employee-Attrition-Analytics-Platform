"""
hierarchical_clustering.py - Agglomerative Hierarchical Clustering, SciPy dendrogram, and cross-comparison.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DEFAULT_K_CLUSTERS, DENDROGRAM_MAX_SAMPLES, RANDOM_STATE
from src.clustering import generate_cluster_profiles


def train_hierarchical_clustering(
    X: np.ndarray,
    n_clusters: int = DEFAULT_K_CLUSTERS,
    linkage_method: str = "ward",
    affinity: str = "euclidean",
) -> Tuple[AgglomerativeClustering, np.ndarray, Dict[str, Any]]:
    """
    Executes Agglomerative Hierarchical Clustering.
    Validates distance/linkage compatibility (Ward requires Euclidean distance).
    """
    if linkage_method == "ward" and affinity not in ["euclidean", "deprecated"]:
        raise ValueError("Ward linkage requires Euclidean distance metric.")

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage_method,
    )
    labels = model.fit_predict(X)

    sil = float(silhouette_score(X, labels))
    db = float(davies_bouldin_score(X, labels))
    ch = float(calinski_harabasz_score(X, labels))

    metrics = {
        "n_clusters": n_clusters,
        "linkage": linkage_method,
        "silhouette_score": round(sil, 4),
        "davies_bouldin_index": round(db, 4),
        "calinski_harabasz_index": round(ch, 2),
    }

    return model, labels, metrics


def compute_scipy_linkage(
    X: np.ndarray,
    method: str = "ward",
    metric: str = "euclidean",
    sample_size: int = DENDROGRAM_MAX_SAMPLES,
    random_state: int = RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes SciPy linkage matrix for dendrogram visualization.
    For performance and visual readability, uses a stratified or random sample if X is large.
    Returns (linkage_matrix, sampled_indices).
    """
    n_total = len(X)
    if n_total > sample_size:
        rng = np.random.default_rng(random_state)
        sampled_idx = rng.choice(n_total, size=sample_size, replace=False)
        X_sample = X[sampled_idx]
    else:
        sampled_idx = np.arange(n_total)
        X_sample = X

    Z = linkage(X_sample, method=method, metric=metric)
    return Z, sampled_idx


def compare_clustering_algorithms(
    kmeans_labels: np.ndarray,
    hierarchical_labels: np.ndarray,
) -> Dict[str, Any]:
    """
    Compares K-Means and Hierarchical Clustering segmentations.
    Computes Adjusted Rand Index (ARI), Normalized Mutual Information (NMI),
    and cross-tabulation contingency matrix.
    """
    ari = float(adjusted_rand_score(kmeans_labels, hierarchical_labels))
    nmi = float(normalized_mutual_info_score(kmeans_labels, hierarchical_labels))

    # Cross-tabulation contingency table
    contingency = pd.crosstab(
        pd.Series(kmeans_labels, name="KMeans_Cluster"),
        pd.Series(hierarchical_labels, name="Hierarchical_Cluster"),
    )

    return {
        "adjusted_rand_index": round(ari, 4),
        "normalized_mutual_info": round(nmi, 4),
        "contingency_matrix": contingency,
        "interpretation": (
            f"Clustering agreement: Adjusted Rand Index = {ari:.4f}, Normalized Mutual Information = {nmi:.4f}. "
            "K-Means builds spherical Voronoi partitions optimizing centroid variance, whereas Agglomerative "
            "Hierarchical builds a bottom-up tree merging clusters based on minimum variance increase (Ward)."
        ),
    }


if __name__ == "__main__":
    from src.data_loader import load_or_generate_dataset
    from src.data_cleaning import run_data_cleaning_pipeline
    from src.feature_engineering import WorkforceFeaturePipeline
    from src.clustering import train_kmeans_model

    raw_df = load_or_generate_dataset()
    cleaned_df, _ = run_data_cleaning_pipeline(raw_df)

    feat_pipe = WorkforceFeaturePipeline()
    X_arr, _ = feat_pipe.fit_transform(cleaned_df)

    # Train K-Means
    _, km_labels, km_metrics = train_kmeans_model(X_arr, n_clusters=4)

    # Train Hierarchical
    h_model, h_labels, h_metrics = train_hierarchical_clustering(X_arr, n_clusters=4)
    print(f"Hierarchical Clustering Metrics (k=4): {h_metrics}")

    # Comparison
    comp = compare_clustering_algorithms(km_labels, h_labels)
    print(f"Adjusted Rand Index: {comp['adjusted_rand_index']}")
    print(f"Normalized Mutual Info: {comp['normalized_mutual_info']}")
    print("\nContingency Matrix (K-Means vs Hierarchical):")
    print(comp["contingency_matrix"])
