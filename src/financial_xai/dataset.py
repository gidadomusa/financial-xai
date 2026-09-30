"""Utilities for generating synthetic financial market data.

This module creates a simple tabular dataset where the target represents a
positive future return event.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def generate_synthetic_market_data(
    n_samples: int = 2000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Generate a synthetic dataset for a binary financial return prediction task.

    Parameters
    ----------
    n_samples : int
        Number of observations.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    X : pandas.DataFrame
        Feature matrix.
    y : pandas.Series
        Binary target where 1 indicates positive future return.
    """
    rng = np.random.default_rng(seed)

    market_return = rng.normal(0.0005, 0.02, n_samples)
    volatility = rng.gamma(shape=2.0, scale=0.01, size=n_samples)
    momentum = rng.normal(0.001, 0.03, n_samples)
    sentiment = rng.normal(0.0, 1.0, n_samples)
    volume = rng.lognormal(mean=3.0, sigma=0.5, size=n_samples)
    spread = rng.uniform(0.001, 0.05, n_samples)
    liquidity = rng.uniform(0.2, 1.0, n_samples)
    sector_beta = rng.normal(1.0, 0.3, n_samples)
    macro_rate = rng.normal(0.02, 0.01, n_samples)

    score = (
        5.0 * market_return
        + 3.0 * momentum
        + 1.5 * sentiment
        + 0.5 * liquidity
        - 2.0 * volatility
        - 1.0 * spread
        + 2.0 * macro_rate
        + 0.8 * sector_beta
    )

    # Convert into a probability and then a binary target.
    probability = 1 / (1 + np.exp(-score))
    target = rng.binomial(1, probability)

    data = pd.DataFrame(
        {
            "market_return": market_return,
            "volatility": volatility,
            "momentum": momentum,
            "sentiment": sentiment,
            "volume": volume,
            "spread": spread,
            "liquidity": liquidity,
            "sector_beta": sector_beta,
            "macro_rate": macro_rate,
        }
    )

    return data, pd.Series(target, name="positive_future_return")
