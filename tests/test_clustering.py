"""
test_clustering.py - Unit tests for K-Means, Hierarchical clustering, and model metrics.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import generate_synthetic_workforce_data
from src.data_cleaning import run_data_cleaning_pipeline
from src.feature_engineering import WorkforceFeaturePipeline
from src.clustering import evaluate_kmeans_candidates, train_kmeans_model, generate_cluster_profiles
from src.hierarchical_clustering import (
    train_hierarchical_clustering,
    compute_scipy_linkage,
    compare_clustering_algorithms,
)


@pytest.fixture
def preprocessed_test_data():
    """Generates a small test dataset and returns transformed matrix and dataframe."""
    df = generate_synthetic_workforce_data(n_samples=120, random_seed=42)
    cleaned, _ = run_data_cleaning_pipeline(df)
    feat_pipe = WorkforceFeaturePipeline()
    X, _ = feat_pipe.fit_transform(cleaned)
    return cleaned, X, feat_pipe


def test_feature_pipeline_scaling_and_encoding(preprocessed_test_data):
    """Verifies that StandardScaler centers numeric features around 0 and OneHotEncoder creates dummy columns."""
    cleaned, X, pipe = preprocessed_test_data
    assert X.shape[0] == 120
    assert X.shape[1] > len(pipe.numerical_cols)  # Categorical features expanded
    # Verify zero-mean property of scaled continuous variables
    num_indices = list(range(len(pipe.numerical_cols)))
    means = np.mean(X[:, num_indices], axis=0)
    np.testing.assert_allclose(means, 0.0, atol=1e-1)


def test_kmeans_training_and_metrics(preprocessed_test_data):
    """Verifies K-Means fitting, label shapes, and metric ranges."""
    _, X, _ = preprocessed_test_data
    k = 3
    model, labels, metrics = train_kmeans_model(X, n_clusters=k, random_state=42)

    assert len(labels) == len(X)
    assert len(np.unique(labels)) == k
    assert -1.0 <= metrics["silhouette_score"] <= 1.0
    assert metrics["davies_bouldin_index"] >= 0.0
    assert metrics["calinski_harabasz_index"] > 0.0
    assert metrics["n_clusters"] == k


def test_kmeans_candidate_evaluation(preprocessed_test_data):
    """Verifies candidate evaluation across K range."""
    _, X, _ = preprocessed_test_data
    eval_df = evaluate_kmeans_candidates(X, k_range=(2, 4), random_state=42)

    assert len(eval_df) == 3
    assert list(eval_df["k"]) == [2, 3, 4]
    assert "Silhouette_Score" in eval_df.columns
    assert "Inertia" in eval_df.columns


def test_hierarchical_clustering_ward(preprocessed_test_data):
    """Verifies Hierarchical Agglomerative clustering with Ward linkage."""
    _, X, _ = preprocessed_test_data
    h_k = 3
    model, labels, metrics = train_hierarchical_clustering(X, n_clusters=h_k, linkage_method="ward")

    assert len(labels) == len(X)
    assert len(np.unique(labels)) == h_k
    assert metrics["linkage"] == "ward"

    # Test SciPy linkage
    Z, sampled = compute_scipy_linkage(X, sample_size=40)
    assert Z.shape[0] == len(sampled) - 1
    assert Z.shape[1] == 4


def test_clustering_comparison(preprocessed_test_data):
    """Verifies cross-comparison ARI and NMI computation."""
    _, X, _ = preprocessed_test_data
    _, km_labels, _ = train_kmeans_model(X, n_clusters=3, random_state=42)
    _, h_labels, _ = train_hierarchical_clustering(X, n_clusters=3)

    comp = compare_clustering_algorithms(km_labels, h_labels)
    assert -1.0 <= comp["adjusted_rand_index"] <= 1.0
    assert 0.0 <= comp["normalized_mutual_info"] <= 1.0
    assert comp["contingency_matrix"].shape == (3, 3)
