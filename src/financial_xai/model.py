"""Dataset utilities for stock and portfolio forecasting."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def create_supervised_dataset(
    features: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    lookback: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a supervised learning dataset from time series features.
    
    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix with dates as index (sorted).
    target_column : str
        Column name for the target variable (e.g., "return_1d").
    test_size : float
        Fraction of data to use for testing.
    lookback : int
        Number of rows to drop at the start (for rolling features to stabilize).
        
    Returns
    -------
    X_train, X_test, y_train, y_test
        Train and test splits.
    """
    # Drop rows with NaN values
    data = features.dropna()
    
    # Drop early rows if lookback is specified
    if lookback > 0:
        data = data.iloc[lookback:].reset_index(drop=True)
    
    # Separate features and target
    y = data[target_column]
    X = data.drop(columns=[target_column])
    
    # Time-based split (no shuffle for time series)
    split_idx = int(len(X) * (1 - test_size))
    
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    return X_train, X_test, y_train, y_test


def create_binary_target(returns: pd.Series, threshold: float = 0.0) -> pd.Series:
    """Convert continuous returns to binary target (positive vs non-positive).
    
    Parameters
    ----------
    returns : pd.Series
        Return values.
    threshold : float
        Threshold for classification (default 0.0 for positive return).
        
    Returns
    -------
    pd.Series
        Binary target (1 if return > threshold, 0 otherwise).
    """
    return (returns > threshold).astype(int)


def generate_synthetic_portfolio_data(
    n_samples: int = 2000,
    n_assets: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data for multiple assets.
    
    Parameters
    ----------
    n_samples : int
        Number of time steps.
    n_assets : int
        Number of assets in the portfolio.
    seed : int
        Random seed.
        
    Returns
    -------
    pd.DataFrame
        OHLCV data with date index.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start="2020-01-01", periods=n_samples, freq="D")
    
    data_list = []
    for asset_id in range(n_assets):
        # Generate price series using geometric Brownian motion
        mu = 0.0005
        sigma = 0.02
        S0 = 100
        
        log_returns = rng.normal(mu, sigma, n_samples)
        prices = S0 * np.exp(np.cumsum(log_returns))
        
        # Generate OHLCV
        daily_vol = rng.uniform(0.005, 0.02, n_samples)
        opens = prices * (1 + rng.normal(0, daily_vol / 4, n_samples))
        closes = prices * (1 + rng.normal(0, daily_vol / 4, n_samples))
        highs = np.maximum(opens, closes) * (1 + daily_vol)
        lows = np.minimum(opens, closes) * (1 - daily_vol)
        volume = rng.lognormal(mean=10, sigma=1, size=n_samples)
        
        asset_data = pd.DataFrame(
            {
                "asset_id": asset_id,
                "date": dates,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volume,
            }
        )
        data_list.append(asset_data)
    
    return pd.concat(data_list, ignore_index=True).sort_values(["asset_id", "date"])
