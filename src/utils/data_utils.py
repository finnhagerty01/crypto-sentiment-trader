# src/utils/data_utils.py
"""
Common data processing utilities used across the pipeline.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)

def validate_dataframe(df: pd.DataFrame, 
                       required_cols: List[str],
                       name: str = "dataframe") -> pd.DataFrame:
    """
    Validate that a DataFrame contains required columns.
    
    Args:
        df: DataFrame to validate
        required_cols: List of required column names
        name: Name for error messages
    
    Returns:
        The validated DataFrame
    
    Raises:
        KeyError: If required columns are missing
    """
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"{name} missing required columns: {missing}")
    return df

def safe_log_return(prices: pd.Series, 
                    shift: int = 1,
                    fillna: float = 0.0) -> pd.Series:
    """
    Calculate log returns safely, handling edge cases.
    
    Args:
        prices: Series of prices
        shift: Number of periods to shift for return calculation
        fillna: Value to fill NaN returns
    
    Returns:
        Series of log returns
    """
    # Ensure no zero or negative prices
    prices_clean = prices.replace([0, -np.inf, np.inf], np.nan)
    
    # Calculate log returns
    returns = np.log(prices_clean / prices_clean.shift(shift))
    
    # Handle infinities that might arise
    returns = returns.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN values
    return returns.fillna(fillna)

def add_technical_features(df: pd.DataFrame,
                          price_col: str = "close",
                          volume_col: str = "volume") -> pd.DataFrame:
    """
    Add technical indicators as features.
    
    Args:
        df: DataFrame with OHLCV data
        price_col: Name of price column
        volume_col: Name of volume column
    
    Returns:
        DataFrame with additional technical features
    """
    df = df.copy()
    
    # Price-based features
    df['ret_1'] = safe_log_return(df[price_col], shift=1)
    df['ret_3'] = df['ret_1'].rolling(3).sum()
    df['ret_6'] = df['ret_1'].rolling(6).sum()
    
    # Volatility
    df['vol_6'] = df['ret_1'].rolling(6).std()
    df['vol_24'] = df['ret_1'].rolling(24).std()
    
    # Volume features
    df['volume_ratio'] = df[volume_col] / df[volume_col].rolling(24).mean()
    
    # Price position within high-low range
    df['price_position'] = (df[price_col] - df['low']) / (df['high'] - df['low'] + 1e-10)
    
    # RSI-like momentum (simplified)
    gains = df['ret_1'].clip(lower=0)
    losses = -df['ret_1'].clip(upper=0)
    avg_gain = gains.rolling(14).mean()
    avg_loss = losses.rolling(14).mean()
    df['rsi_14'] = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-10)))
    
    return df

def calculate_sharpe_ratio(returns: pd.Series,
                          periods_per_year: int = 252) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Args:
        returns: Series of returns
        periods_per_year: Number of periods in a year (252 for daily, 365*24 for hourly)
    
    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0
    
    mean_return = returns.mean()
    std_return = returns.std()
    
    if std_return == 0:
        return 0.0
    
    return np.sqrt(periods_per_year) * mean_return / std_return

def calculate_max_drawdown(cumulative_returns: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    """
    Calculate maximum drawdown and when it occurred.
    
    Args:
        cumulative_returns: Series of cumulative returns
    
    Returns:
        Tuple of (max_drawdown, peak_date, trough_date)
    """
    cumsum = cumulative_returns.cumsum()
    running_max = cumsum.expanding().max()
    drawdown = cumsum - running_max
    
    max_dd = drawdown.min()
    if max_dd == 0:
        return 0.0, pd.NaT, pd.NaT
    
    trough_date = drawdown.idxmin()
    peak_date = cumsum[:trough_date].idxmax()
    
    return max_dd, peak_date, trough_date