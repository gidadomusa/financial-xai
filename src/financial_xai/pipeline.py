"""Explainability utilities using SHAP for portfolio forecasting models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


def compute_shap_values(model: Any, X: pd.DataFrame | np.ndarray, max_samples: int = 100) -> np.ndarray:
    """Compute SHAP values for model predictions.
    
    Parameters
    ----------
    model : model object
        Fitted model with underlying sklearn estimator.
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    max_samples : int
        Maximum number of samples for SHAP background dataset.
        
    Returns
    -------
    np.ndarray
        SHAP values.
    """
    if not HAS_SHAP:
        raise ImportError("SHAP is not installed. Install it with: pip install shap")
    
    # Use a subset for the background dataset if needed
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = X
    
    if len(X_array) > max_samples:
        background_indices = np.random.choice(len(X_array), max_samples, replace=False)
        X_background = X_array[background_indices]
    else:
        X_background = X_array
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model.model)
    X_scaled = model.scaler.transform(X_array)
    shap_values = explainer.shap_values(X_scaled)
    
    # Handle binary classification
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    return shap_values


def feature_importance_from_shap(
    shap_values: np.ndarray,
    feature_names: list[str] | None = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """Compute feature importance from SHAP values.
    
    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values (n_samples x n_features).
    feature_names : list[str], optional
        Names of features.
    top_n : int
        Number of top features to return.
        
    Returns
    -------
    pd.DataFrame
        Top features ranked by mean absolute SHAP value.
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(mean_abs_shap))]
    
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap,
        }
    ).sort_values("mean_abs_shap", ascending=False)
    
    return importance_df.head(top_n).reset_index(drop=True)


def explain_prediction(
    shap_values: np.ndarray,
    X: pd.DataFrame | np.ndarray,
    feature_names: list[str] | None = None,
    sample_idx: int = 0,
    top_n: int = 5,
) -> pd.DataFrame:
    """Explain a single prediction using SHAP values.
    
    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values.
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    feature_names : list[str], optional
        Names of features.
    sample_idx : int
        Index of the sample to explain.
    top_n : int
        Number of top contributing features to show.
        
    Returns
    -------
    pd.DataFrame
        Top contributing features for the prediction.
    """
    if isinstance(X, pd.DataFrame):
        X_array = X.values
        if feature_names is None:
            feature_names = list(X.columns)
    else:
        X_array = X
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X_array.shape[1])]
    
    sample_shap = shap_values[sample_idx]
    sample_features = X_array[sample_idx]
    
    explanation_df = pd.DataFrame(
        {
            "feature": feature_names,
            "value": sample_features,
            "shap_value": sample_shap,
        }
    )
    
    explanation_df["abs_shap"] = np.abs(explanation_df["shap_value"])
    explanation_df = explanation_df.sort_values("abs_shap", ascending=False)
    
    return explanation_df.head(top_n)[["feature", "value", "shap_value"]].reset_index(drop=True)
