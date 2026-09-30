"""Model training utilities for explainable finance experiments."""

from __future__ import annotations

from typing import Any

import numpy as np


class BaseFinancialModel:
    """Base class placeholder for training and prediction."""

    def __init__(self, model: Any = None):
        self.model = model

    def fit(self, X: np.ndarray, y: np.ndarray) -> Any:
        if self.model is None:
            raise ValueError("No model instance provided.")
        self.model.fit(X, y)
        return self.model

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("No model instance provided.")
        return self.model.predict(X)
