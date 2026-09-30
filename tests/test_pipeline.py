"""Pipeline orchestration for the financial XAI demo."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .dataset import generate_synthetic_market_data
from .explainability import explain_top_features
from .model import summarize_model_metrics, train_evaluation_model


def run_financial_xai_demo() -> tuple[dict[str, float], pd.DataFrame]:
    """Run the full demo pipeline and return metrics and top features."""
    features, target = generate_synthetic_market_data(n_samples=2000, seed=42)
    metrics = train_evaluation_model(features, target)
    summary = summarize_model_metrics(metrics)
    top_features = explain_top_features(metrics["model"], metrics["X_test"])
    return summary, top_features
