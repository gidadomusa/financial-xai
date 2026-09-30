"""Main pipeline entry point for a financial XAI demo."""

from __future__ import annotations

from .dataset import generate_synthetic_market_data
from .explainability import explain_top_features
from .pipeline import run_financial_xai_demo


def main() -> None:
    """Generate sample data, train the model, and show feature importance."""
    metrics, feature_importance = run_financial_xai_demo()
    print("\nFinancial XAI demo summary")
    print("=" * 28)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print("\nTop contributing features:")
    print(feature_importance.to_string(index=False))


if __name__ == "__main__":
    main()
