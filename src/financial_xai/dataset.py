"""Feature engineering for stock and portfolio forecasting.

This module provides utilities to create financial features from raw OHLCV data,
including returns, volatility, momentum, and liquidity metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_returns(prices: pd.Series, periods: int | list[int] = [1, 5, 20]) -> pd.DataFrame:
    """Compute returns over multiple time horizons.
    
    Parameters
    ----------
    prices : pd.Series
        Price series (typically close prices).
    periods : int or list[int]
        Number of periods for return calculation.
        
    Returns
    -------
    pd.DataFrame
        Returns computed over the specified periods.
    """
    if isinstance(periods, int):
        periods = [periods]
    
    returns = pd.DataFrame(index=prices.index)
    for p in periods:
        returns[f"return_{p}d"] = prices.pct_change(p)
    
    return returns


def compute_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """Compute rolling volatility (standard deviation of returns).
    
    Parameters
    ----------
    returns : pd.Series
        Daily or periodic returns.
    window : int
        Rolling window size in periods.
        
    Returns
    -------
    pd.Series
        Rolling volatility.
    """
    return returns.rolling(window=window).std()


def compute_momentum(returns: pd.Series, window: int = 20) -> pd.Series:
    """Compute momentum as cumulative return over a window.
    
    Parameters
    ----------
    returns : pd.Series
        Daily or periodic returns.
    window : int
        Rolling window size in periods.
        
    Returns
    -------
    pd.Series
        Cumulative momentum over the window.
    """
    return (1 + returns).rolling(window=window).prod() - 1


def compute_rsi(returns: pd.Series, window: int = 14) -> pd.Series:
    """Compute Relative Strength Index (RSI).
    
    Parameters
    ----------
    returns : pd.Series
        Daily or periodic returns.
    window : int
        RSI window.
        
    Returns
    -------
    pd.Series
        RSI values between 0 and 100.
    """
    gains = returns.clip(lower=0)
    losses = -returns.clip(upper=0)
    
    avg_gain = gains.rolling(window=window).mean()
    avg_loss = losses.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def compute_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Compute MACD (Moving Average Convergence Divergence).
    
    Parameters
    ----------
    prices : pd.Series
        Price series.
    fast : int
        Fast EMA period.
    slow : int
        Slow EMA period.
    signal : int
        Signal line EMA period.
        
    Returns
    -------
    pd.DataFrame
        MACD line, signal line, and histogram.
    """
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    
    return pd.DataFrame(
        {
            "macd": macd_line,
            "signal": signal_line,
            "histogram": histogram,
        },
        index=prices.index,
    )


def compute_volume_features(volume: pd.Series, close: pd.Series, high: pd.Series, low: pd.Series, window: int = 20) -> pd.DataFrame:
    """Compute volume-based features.
    
    Parameters
    ----------
    volume : pd.Series
        Trading volume.
    close : pd.Series
        Close price.
    high : pd.Series
        High price.
    low : pd.Series
        Low price.
    window : int
        Rolling window.
        
    Returns
    -------
    pd.DataFrame
        Volume-based features (on-balance volume, volume ratio, etc.).
    """
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    avg_volume = volume.rolling(window=window).mean()
    volume_ratio = volume / avg_volume
    
    # Money flow index components
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    positive_mf = money_flow.where(close.diff() > 0, 0)
    negative_mf = money_flow.where(close.diff() < 0, 0)
    
    return pd.DataFrame(
        {
            "obv": obv,
            "volume_ratio": volume_ratio,
            "positive_money_flow": positive_mf.rolling(window=window).sum(),
            "negative_money_flow": negative_mf.rolling(window=window).sum(),
        },
        index=volume.index,
    )


def compute_bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Compute Bollinger Bands.
    
    Parameters
    ----------
    prices : pd.Series
        Price series.
    window : int
        Moving average window.
    num_std : float
        Number of standard deviations.
        
    Returns
    -------
    pd.DataFrame
        Upper band, middle band, lower band, and bandwidth.
    """
    sma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    
    upper = sma + num_std * std
    lower = sma - num_std * std
    bandwidth = (upper - lower) / sma
    
    return pd.DataFrame(
        {
            "bb_upper": upper,
            "bb_middle": sma,
            "bb_lower": lower,
            "bb_bandwidth": bandwidth,
        },
        index=prices.index,
    )


def engineer_features(df: pd.DataFrame, target_periods: list[int] = [1, 5, 20]) -> pd.DataFrame:
    """Comprehensive feature engineering pipeline for stock forecasting.
    
    Parameters
    ----------
    df : pd.DataFrame
        Market data with columns: open, high, low, close, volume.
    target_periods : list[int]
        Periods for forward-looking return targets.
        
    Returns
    -------
    pd.DataFrame
        Features and forward returns as target variable.
    """
    features = df.copy()
    
    # Returns
    returns_df = compute_returns(features["close"], periods=target_periods)
    features = pd.concat([features, returns_df], axis=1)
    
    # Volatility
    daily_returns = features["close"].pct_change()
    features["volatility_20d"] = compute_volatility(daily_returns, window=20)
    features["volatility_60d"] = compute_volatility(daily_returns, window=60)
    
    # Momentum
    features["momentum_20d"] = compute_momentum(daily_returns, window=20)
    features["momentum_60d"] = compute_momentum(daily_returns, window=60)
    
    # RSI
    features["rsi_14"] = compute_rsi(daily_returns, window=14)
    
    # MACD
    macd_df = compute_macd(features["close"])
    features = pd.concat([features, macd_df], axis=1)
    
    # Volume features
    volume_df = compute_volume_features(
        features["volume"],
        features["close"],
        features["high"],
        features["low"],
        window=20,
    )
    features = pd.concat([features, volume_df], axis=1)
    
    # Bollinger Bands
    bb_df = compute_bollinger_bands(features["close"], window=20)
    features = pd.concat([features, bb_df], axis=1)
    
    # Price-to-band position
    features["bb_position"] = (
        (features["close"] - features["bb_lower"]) /
        (features["bb_upper"] - features["bb_lower"])
    )
    
    return features
