"""
feature_engineering.py - Preprocessing pipelines, scaling, encoding, and artifact persistence.
"""

import sys
from pathlib import Path
from typing import Tuple, List
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    PREPROCESSOR_FILE,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    EXCLUDED_FEATURES,
)


class WorkforceFeaturePipeline:
    """
    Manages end-to-end feature selection, scaling (StandardScaler),
    and categorical encoding (OneHotEncoder) for unsupervised clustering.
    """

    def __init__(
        self,
        numerical_cols: List[str] = None,
        categorical_cols: List[str] = None,
        excluded_cols: List[str] = None,
    ):
        self.numerical_cols = numerical_cols or NUMERICAL_FEATURES
        self.categorical_cols = categorical_cols or CATEGORICAL_FEATURES
        self.excluded_cols = excluded_cols or EXCLUDED_FEATURES
        self.pipeline: ColumnTransformer = None
        self.transformed_feature_names: List[str] = []
        self.fitted_ = False

    def _select_available_features(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Filters requested features against available columns in the input dataframe."""
        valid_num = [c for c in self.numerical_cols if c in df.columns and c not in self.excluded_cols]
        valid_cat = [c for c in self.categorical_cols if c in df.columns and c not in self.excluded_cols]
        return valid_num, valid_cat

    def build_transformer(self, valid_num: List[str], valid_cat: List[str]) -> ColumnTransformer:
        """Constructs a scikit-learn ColumnTransformer."""
        transformers = []
        if valid_num:
            # StandardScaler scales each feature to 0 mean and unit variance.
            # Essential for distance-based clustering (Euclidean distance) so high-magnitude
            # variables (e.g. MonthlyIncome) do not swamp small-scale variables (e.g. JobSatisfaction).
            transformers.append(("num", StandardScaler(), valid_num))

        if valid_cat:
            # OneHotEncoder transforms discrete nominal categories into binary indicator vectors.
            transformers.append((
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                valid_cat,
            ))

        return ColumnTransformer(transformers=transformers, remainder="drop")

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Fits preprocessing transformations on employee dataframe and transforms it.
        Returns dense array X and labeled DataFrame.
        """
        valid_num, valid_cat = self._select_available_features(df)
        self.pipeline = self.build_transformer(valid_num, valid_cat)

        X_array = self.pipeline.fit_transform(df)

        # Retrieve output feature names
        feature_names = []
        if valid_num:
            feature_names.extend(valid_num)
        if valid_cat:
            cat_encoder = self.pipeline.named_transformers_["cat"]
            encoded_names = list(cat_encoder.get_feature_names_out(valid_cat))
            feature_names.extend(encoded_names)

        self.transformed_feature_names = feature_names
        self.fitted_ = True

        X_df = pd.DataFrame(X_array, columns=feature_names, index=df.index)
        return X_array, X_df

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """Transforms new employee data using the previously fitted preprocessor."""
        if not self.fitted_ or self.pipeline is None:
            raise RuntimeError("Pipeline must be fitted before calling transform().")

        X_array = self.pipeline.transform(df)
        X_df = pd.DataFrame(X_array, columns=self.transformed_feature_names, index=df.index)
        return X_array, X_df

    def save(self, filepath: Path = PREPROCESSOR_FILE) -> None:
        """Persists fitted pipeline to disk."""
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: Path = PREPROCESSOR_FILE) -> "WorkforceFeaturePipeline":
        """Loads fitted pipeline from disk."""
        return joblib.load(filepath)


if __name__ == "__main__":
    from src.data_cleaning import run_data_cleaning_pipeline
    from src.data_loader import load_or_generate_dataset

    raw_df = load_or_generate_dataset()
    cleaned_df, _ = run_data_cleaning_pipeline(raw_df)

    feat_pipe = WorkforceFeaturePipeline()
    X_arr, X_df = feat_pipe.fit_transform(cleaned_df)
    feat_pipe.save()

    print(f"Transformed feature matrix shape: {X_arr.shape}")
    print(f"Total features created: {len(feat_pipe.transformed_feature_names)}")
    print(f"Sample transformed columns: {feat_pipe.transformed_feature_names[:8]}")
