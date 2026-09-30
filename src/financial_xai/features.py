"""Utilities for loading and validating financial market data."""

from pathlib import Path
from typing import Optional

import pandas as pd


def load_market_data(path: str | Path, date_column: str = "date") -> pd.DataFrame:
    """Load market data from CSV or Parquet file.
    
    Parameters
    ----------
    path : str or Path
        Path to the data file (CSV or Parquet).
    date_column : str
        Name of the date column. Will be converted to datetime.
        
    Returns
    -------
    pd.DataFrame
        Loaded data with datetime index.
    """
    path = Path(path)
    
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix in [".parquet", ".pq"]:
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column).reset_index(drop=True)
    
    return df


def validate_market_data(df: pd.DataFrame, required_columns: list[str]) -> bool:
    """Validate that market data has required columns and no critical missing values.
    
    Parameters
    ----------
    df : pd.DataFrame
        The market data to validate.
    required_columns : list[str]
        List of required column names.
        
    Returns
    -------
    bool
        True if data is valid.
        
    Raises
    ------
    ValueError
        If data is missing required columns or has too many missing values.
    """
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    missing_pct = df[required_columns].isnull().sum() / len(df)
    if (missing_pct > 0.1).any():
        raise ValueError(f"Columns with >10% missing values: {missing_pct[missing_pct > 0.1].index.tolist()}")
    
    return True


def load_and_validate(path: str | Path, required_columns: list[str]) -> pd.DataFrame:
    """Load and validate market data in one step.
    
    Parameters
    ----------
    path : str or Path
        Path to the data file.
    required_columns : list[str]
        List of required column names.
        
    Returns
    -------
    pd.DataFrame
        Validated market data.
    """
    df = load_market_data(path)
    validate_market_data(df, required_columns)
    return df
