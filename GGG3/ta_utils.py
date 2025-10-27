# ta_utils.py
import pandas as pd

def sma(series: pd.Series, n: int) -> pd.Series:
    """Simple Moving Average"""
    return series.rolling(n, min_periods=1).mean()

def stdev(series: pd.Series, n: int) -> pd.Series:
    """Standard Deviation"""
    return series.rolling(n, min_periods=1).std(ddof=0)

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, 1e-12)
    return 100 - (100 / (1 + rs))