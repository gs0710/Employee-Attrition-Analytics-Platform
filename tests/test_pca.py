"""
test_pca.py - Unit tests for Principal Component Analysis, variance decomposition, and loadings.
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
from src.pca_analysis import WorkforcePCAModel


@pytest.fixture
def test_feature_matrix():
    df = generate_synthetic_workforce_data(n_samples=100, random_seed=42)
    cleaned, _ = run_data_cleaning_pipeline(df)
    pipe = WorkforceFeaturePipeline()
    X, _ = pipe.fit_transform(cleaned)
    return X, pipe.transformed_feature_names


def test_pca_fit_transform_dimensions(test_feature_matrix):
    """Verifies that PCA transforms the feature matrix and computes explained variance."""
    X, feature_names = test_feature_matrix
    pca_model = WorkforcePCAModel()
    X_pca, results = pca_model.fit_transform(X, feature_names)

    assert X_pca.shape[0] == X.shape[0]
    assert X_pca.shape[1] == X.shape[1]

    var_sum = results["variance_summary"]
    assert 0 < var_sum["pc1_explained_variance"] <= 100
    assert 0 < var_sum["pc2_explained_variance"] <= 100
    assert var_sum["pc1_pc2_cumulative_variance"] == round(var_sum["pc1_explained_variance"] + var_sum["pc2_explained_variance"], 2)

    # Monotonic accumulation of cumulative variance
    cum_var = results["cumulative_variance_ratio"]
    assert np.all(np.diff(cum_var) >= 0)
    assert np.isclose(cum_var[-1], 1.0, atol=1e-3)


def test_pca_thresholds_computation(test_feature_matrix):
    """Verifies calculation of components required for 80%, 90%, and 95% variance."""
    X, feature_names = test_feature_matrix
    pca_model = WorkforcePCAModel()
    _, results = pca_model.fit_transform(X, feature_names)

    var_sum = results["variance_summary"]
    c80 = var_sum["components_for_80_pct"]
    c90 = var_sum["components_for_90_pct"]
    c95 = var_sum["components_for_95_pct"]

    assert 1 <= c80 <= c90 <= c95 <= X.shape[1]


def test_pca_loadings_and_top_contributors(test_feature_matrix):
    """Verifies extraction of component loadings and top loaded attributes."""
    X, feature_names = test_feature_matrix
    pca_model = WorkforcePCAModel()
    pca_model.fit_transform(X, feature_names)

    loadings_df = pca_model.get_loadings_dataframe(n_top_components=3)
    assert loadings_df.shape == (len(feature_names), 3)

    top_pc1 = pca_model.get_top_contributors(component_idx=0, top_n=5)
    assert len(top_pc1) == 5
    assert "Feature" in top_pc1.columns
    assert "Loading" in top_pc1.columns
    assert "Absolute_Loading" in top_pc1.columns
