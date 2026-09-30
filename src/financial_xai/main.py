"""End-to-end pipeline for stock and portfolio forecasting."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .dataset import create_supervised_dataset, create_binary_target, generate_synthetic_portfolio_data
from .features import engineer_features
from .model import compare_models, evaluate_model, PortfolioModel
from .explainability import compute_shap_values, feature_importance_from_shap


def run_portfolio_forecasting_pipeline(
    data: pd.DataFrame | None = None,
    target_return_period: int = 5,
    model_type: str = "random_forest",
    return_explanation: bool = True,
) -> dict[str, Any]:
    """Run the complete portfolio forecasting and explanation pipeline.
    
    Parameters
    ----------
    data : pd.DataFrame, optional
        Market data. If None, synthetic data is generated.
    target_return_period : int
        Number of days for forward return target.
    model_type : str
        Type of model to train.
    return_explanation : bool
        Whether to compute SHAP explanations.
        
    Returns
    -------
    dict[str, Any]
        Results including model, metrics, and explanations.
    """
    # Step 1: Generate or load data
    if data is None:
        print("Generating synthetic portfolio data...")
        raw_data = generate_synthetic_portfolio_data(n_samples=2000, n_assets=3)
    else:
        raw_data = data
    
    # Step 2: Feature engineering
    print("Engineering features...")
    features_all = []
    for asset_id in raw_data["asset_id"].unique():
        asset_data = raw_data[raw_data["asset_id"] == asset_id].copy()
        asset_data = asset_data.set_index("date")
        engineered = engineer_features(asset_data, target_periods=[target_return_period])
        engineered["asset_id"] = asset_id
        features_all.append(engineered)
    
    features_df = pd.concat(features_all, ignore_index=False).reset_index(drop=True)
    
    # Step 3: Create binary target
    print("Creating binary targets...")
    target_col = f"return_{target_return_period}d"
    features_df["target"] = create_binary_target(features_df[target_col], threshold=0.0)
    
    # Step 4: Create supervised dataset
    print("Creating train/test splits...")
    X_train, X_test, y_train, y_test = create_supervised_dataset(
        features_df.drop(columns=[target_col, "asset_id"]),
        target_column="target",
        test_size=0.2,
        lookback=60,
    )
    
    # Step 5: Train model
    print(f"Training {model_type} model...")
    model = PortfolioModel(model_type=model_type, random_state=42)
    model.fit(X_train, y_train)
    
    # Step 6: Evaluate
    print("Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    
    # Step 7: Explain (if requested)
    explanation = None
    feature_importance = None
    if return_explanation:
        print("Computing SHAP explanations...")
        try:
            shap_values = compute_shap_values(model, X_test, max_samples=100)
            feature_importance = feature_importance_from_shap(
                shap_values,
                feature_names=list(X_test.columns),
                top_n=10,
            )
            explanation = {
                "shap_values": shap_values,
                "feature_importance": feature_importance,
            }
        except ImportError:
            print("SHAP not installed. Skipping explanations.")
    
    return {
        "model": model,
        "metrics": metrics,
        "X_test": X_test,
        "y_test": y_test,
        "explanation": explanation,
        "feature_importance": feature_importance,
    }
