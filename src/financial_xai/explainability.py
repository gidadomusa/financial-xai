"""Model training and evaluation for portfolio forecasting."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler


class PortfolioModel:
    """Base class for portfolio forecasting models."""
    
    def __init__(self, model_type: str = "random_forest", **kwargs):
        """Initialize a model.
        
        Parameters
        ----------
        model_type : str
            Type of model: "logistic", "random_forest", or "gradient_boosting".
        **kwargs
            Additional parameters for the model.
        """
        if model_type == "logistic":
            self.model = LogisticRegression(**kwargs)
        elif model_type == "random_forest":
            self.model = RandomForestClassifier(**kwargs)
        elif model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> PortfolioModel:
        """Fit the model.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix.
        y : pd.Series or np.ndarray
            Target variable.
            
        Returns
        -------
        self
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix.
            
        Returns
        -------
        np.ndarray
            Predictions.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict probabilities.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix.
            
        Returns
        -------
        np.ndarray
            Class probabilities.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


def evaluate_model(
    model: PortfolioModel,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
) -> dict[str, float]:
    """Evaluate model performance.
    
    Parameters
    ----------
    model : PortfolioModel
        Fitted model.
    X_test : pd.DataFrame or np.ndarray
        Test features.
    y_test : pd.Series or np.ndarray
        Test targets.
        
    Returns
    -------
    dict[str, float]
        Evaluation metrics.
    """
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    
    cm = confusion_matrix(y_test, predictions)
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }
    
    return metrics


def compare_models(
    X_train: pd.DataFrame | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    y_test: pd.Series | np.ndarray,
) -> dict[str, Any]:
    """Train and compare multiple models.
    
    Parameters
    ----------
    X_train, X_test : pd.DataFrame or np.ndarray
        Training and test features.
    y_train, y_test : pd.Series or np.ndarray
        Training and test targets.
        
    Returns
    -------
    dict[str, Any]
        Results for each model type.
    """
    results = {}
    
    model_configs = [
        ("logistic_regression", {"max_iter": 1000, "random_state": 42}),
        ("random_forest", {"n_estimators": 300, "max_depth": 8, "random_state": 42}),
        ("gradient_boosting", {"n_estimators": 300, "learning_rate": 0.1, "random_state": 42}),
    ]
    
    for model_name, params in model_configs:
        model_type = model_name.split("_")[0]
        model = PortfolioModel(model_type=model_type, **params)
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        results[model_name] = {
            "model": model,
            "metrics": metrics,
        }
    
    return results
