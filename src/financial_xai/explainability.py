"""Explainability utilities for financial model interpretation."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_feature_importance(model: Any, X: np.ndarray) -> np.ndarray:
    """Placeholder feature importance helper."""
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    return np.ones(X.shape[1])
