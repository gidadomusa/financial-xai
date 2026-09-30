"""Utilities for loading input data."""

from pathlib import Path
from typing import Any

import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    return pd.read_csv(path)


def load_parquet(path: str | Path) -> pd.DataFrame:
    """Load a parquet file into a pandas DataFrame."""
    return pd.read_parquet(path)


def validate_data(df: pd.DataFrame) -> bool:
    """Simple validation helper for input datasets."""
    if df.empty:
        return False
    return True
