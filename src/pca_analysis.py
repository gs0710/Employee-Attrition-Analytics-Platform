"""
pca_analysis.py - Principal Component Analysis implementation, variance profiling, loadings, and 2D projections.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import PCA_MODEL_FILE, RANDOM_STATE, TABLES_DIR


class WorkforcePCAModel:
    """
    Fits Principal Component Analysis on standardized workforce feature matrix.
    Computes scree data, cumulative variance, leading feature loadings, and 2D projections.
    """

    def __init__(self, n_components: int = None, random_state: int = RANDOM_STATE):
        self.n_components = n_components
        self.random_state = random_state
        self.pca: PCA = None
        self.feature_names: List[str] = []
        self.fitted_ = False

    def fit_transform(
        self,
        X: np.ndarray,
        feature_names: List[str] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Fits PCA on input matrix X and transforms to principal components.
        Computes explained variance, cumulative variance, and variance thresholds (80%, 90%, 95%).
        """
        max_possible = min(X.shape[0], X.shape[1])
        n_comp = self.n_components if (self.n_components and self.n_components <= max_possible) else max_possible

        self.pca = PCA(n_components=n_comp, random_state=self.random_state)
        X_pca = self.pca.fit_transform(X)

        self.feature_names = feature_names or [f"Feature_{i}" for i in range(X.shape[1])]
        self.fitted_ = True

        explained_var = self.pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)

        # Find components required for 80%, 90%, 95% variance
        comp_80 = int(np.argmax(cumulative_var >= 0.80) + 1) if (cumulative_var >= 0.80).any() else n_comp
        comp_90 = int(np.argmax(cumulative_var >= 0.90) + 1) if (cumulative_var >= 0.90).any() else n_comp
        comp_95 = int(np.argmax(cumulative_var >= 0.95) + 1) if (cumulative_var >= 0.95).any() else n_comp

        variance_summary = {
            "total_input_features": X.shape[1],
            "total_components": n_comp,
            "pc1_explained_variance": round(float(explained_var[0]) * 100, 2),
            "pc2_explained_variance": round(float(explained_var[1]) * 100, 2) if n_comp > 1 else 0.0,
            "pc1_pc2_cumulative_variance": round(float(cumulative_var[1]) * 100, 2) if n_comp > 1 else round(float(cumulative_var[0]) * 100, 2),
            "components_for_80_pct": comp_80,
            "components_for_90_pct": comp_90,
            "components_for_95_pct": comp_95,
        }

        # Build explained variance dataframe
        var_df = pd.DataFrame({
            "Component": [f"PC{i+1}" for i in range(n_comp)],
            "Explained_Variance_Ratio": np.round(explained_var, 4),
            "Explained_Variance_Pct": np.round(explained_var * 100, 2),
            "Cumulative_Variance_Pct": np.round(cumulative_var * 100, 2),
        })

        # Save model and variance table
        joblib.dump(self, PCA_MODEL_FILE)
        var_df.to_csv(TABLES_DIR / "pca_explained_variance.csv", index=False)

        return X_pca, {
            "variance_summary": variance_summary,
            "variance_table": var_df,
            "explained_variance_ratio": explained_var,
            "cumulative_variance_ratio": cumulative_var,
        }

    def get_loadings_dataframe(self, n_top_components: int = 5) -> pd.DataFrame:
        """
        Returns feature loadings matrix for leading components.
        Loadings indicate the correlation between each original feature and the principal component.
        """
        if not self.fitted_ or self.pca is None:
            raise RuntimeError("PCA model is not fitted.")

        n_comp = min(n_top_components, self.pca.n_components_)
        loadings = self.pca.components_[:n_comp].T

        col_names = [f"PC{i+1}" for i in range(n_comp)]
        loadings_df = pd.DataFrame(
            loadings,
            columns=col_names,
            index=self.feature_names,
        )

        loadings_df.to_csv(TABLES_DIR / "pca_component_loadings.csv")
        return loadings_df

    def get_top_contributors(self, component_idx: int = 0, top_n: int = 8) -> pd.DataFrame:
        """Returns the top contributing positive and negative features for a given principal component."""
        if not self.fitted_:
            raise RuntimeError("PCA model is not fitted.")

        comp_loadings = self.pca.components_[component_idx]
        abs_loadings = np.abs(comp_loadings)
        top_indices = np.argsort(abs_loadings)[::-1][:top_n]

        top_df = pd.DataFrame({
            "Feature": [self.feature_names[i] for i in top_indices],
            "Loading": np.round([comp_loadings[i] for i in top_indices], 4),
            "Absolute_Loading": np.round([abs_loadings[i] for i in top_indices], 4),
            "Direction": ["Positive" if comp_loadings[i] >= 0 else "Negative" for i in top_indices],
        })
        return top_df

    @staticmethod
    def load(filepath: Path = PCA_MODEL_FILE) -> "WorkforcePCAModel":
        """Loads fitted PCA model artifact."""
        return joblib.load(filepath)


if __name__ == "__main__":
    from src.data_loader import load_or_generate_dataset
    from src.data_cleaning import run_data_cleaning_pipeline
    from src.feature_engineering import WorkforceFeaturePipeline

    raw_df = load_or_generate_dataset()
    cleaned_df, _ = run_data_cleaning_pipeline(raw_df)

    feat_pipe = WorkforceFeaturePipeline()
    X_arr, _ = feat_pipe.fit_transform(cleaned_df)

    pca_model = WorkforcePCAModel()
    X_pca, pca_results = pca_model.fit_transform(X_arr, feat_pipe.transformed_feature_names)

    print("PCA Variance Summary:")
    print(pca_results["variance_summary"])

    loadings_df = pca_model.get_loadings_dataframe(n_top_components=4)
    print("\nTop Features for PC1:")
    print(pca_model.get_top_contributors(component_idx=0, top_n=5))
    print("\nTop Features for PC2:")
    print(pca_model.get_top_contributors(component_idx=1, top_n=5))
