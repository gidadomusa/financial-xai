"""Explainability utilities using SHAP.

This module produces feature attribution values to interpret model predictions
for a financial classification problem.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap


def compute_shap_values(model: Any, X: pd.DataFrame) -> Any:
    """Compute SHAP values for a fitted model on a feature matrix."""
    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(X)


def explain_top_features(model: Any, X: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Return the most important features based on mean absolute SHAP values."""
    shap_values = compute_shap_values(model, X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    importance = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame(
        {
            "feature": X.columns,
            "mean_abs_shap": importance,
        }
    ).sort_values("mean_abs_shap", ascending=False)

    return feature_importance.head(top_n).reset_index(drop=True)
