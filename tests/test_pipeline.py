"""Main application entry point for portfolio forecasting."""

from .pipeline import run_portfolio_forecasting_pipeline


def main() -> None:
    """Run the portfolio forecasting pipeline."""
    print("Starting Financial XAI - Stock & Portfolio Forecasting")
    print("=" * 50)
    
    results = run_portfolio_forecasting_pipeline(
        data=None,
        target_return_period=5,
        model_type="random_forest",
        return_explanation=True,
    )
    
    # Print results
    print("\nModel Performance Metrics")
    print("-" * 50)
    metrics = results["metrics"]
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {metrics['true_positives']}")
    print(f"  True Negatives:  {metrics['true_negatives']}")
    print(f"  False Positives: {metrics['false_positives']}")
    print(f"  False Negatives: {metrics['false_negatives']}")
    
    if results["feature_importance"] is not None:
        print("\nTop 10 Most Important Features (by SHAP)")
        print("-" * 50)
        print(results["feature_importance"].to_string(index=False))
    else:
        print("\nFeature importance could not be computed.")
    
    print("\n" + "=" * 50)
    print("Portfolio forecasting pipeline completed.")


if __name__ == "__main__":
    main()
