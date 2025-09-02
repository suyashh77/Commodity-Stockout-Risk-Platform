# data_processing.py
"""
Data-fetching and feature engineering for commodity risk index.
Functions are pure and safe to import (no top-level side effects).
"""

from typing import Tuple
import pandas as pd
import numpy as np
import yfinance as yf


COMMODITY_MAP = {
    "Lithium": "LIT",
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Steel": "SLX",
    "Copper": "HG=F",
    "Aluminium": "ALI=F"
}



def fetch_price_data(ticker, start, end):
    """
    Fetch OHLCV data for a ticker. Handles tickers with or without 'Adj Close'.
    """
    df = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False  # ensures Adj Close exists if available
    )
    
    # List of expected columns
    cols = ["Open", "High", "Low", "Close", "Volume"]
    if "Adj Close" in df.columns:
        cols.insert(4, "Adj Close")  # keep Adj Close if it exists
    
    df = df[cols].ffill().dropna()
    return df


def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Compute Average True Range (ATR).
    ATR uses high, low, prev close.
    """
    high = df["High"]
    low = df["Low"]
    prev_close = df["Adj Close"].shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    return atr


def compute_momentum(df: pd.DataFrame, period: int = 30) -> pd.Series:
    """Rate of change over `period` days on Adj Close."""
    return df["Adj Close"].pct_change(periods=period)


def compute_rolling_volatility(df: pd.DataFrame, window: int = 30) -> pd.Series:
    """Rolling std of percent returns (volatility)."""
    returns = df["Adj Close"].pct_change()
    return returns.rolling(window).std()


def compute_rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Simple RSI implementation."""
    delta = df["Adj Close"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window=window).mean()
    ma_down = down.rolling(window=window).mean()
    rs = ma_up / (ma_down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Return MACD line and MACD signal line."""
    exp1 = df["Adj Close"].ewm(span=fast, adjust=False).mean()
    exp2 = df["Adj Close"].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"MACD": macd, "MACD_Signal": macd_signal})


def compute_volume_surge(df: pd.DataFrame, window: int = 7) -> pd.Series:
    """Rolling percent change in volume, smoothed by window."""
    vol_pct = df["Volume"].pct_change()
    return vol_pct.rolling(window).mean()


def detect_shock(df: pd.DataFrame, pct_change_window: int = 5, threshold: float = 0.10) -> pd.Series:
    """
    Binary series that flags when Adj Close increases by > threshold within pct_change_window days.
    (You can tune threshold / window per commodity.)
    """
    shock = (df["Adj Close"].pct_change(periods=pct_change_window) > threshold).astype(int)
    return shock


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds all engineered features to a copy of df and returns the enriched DataFrame.
    Columns added:
        - ATR, Volatility, Momentum, RSI, MACD, MACD_Signal, Volume_Surge, Shock
    """
    df2 = df.copy()
    df2["Volatility"] = compute_rolling_volatility(df2)
    df2["ATR"] = compute_atr(df2)
    df2["Momentum"] = compute_momentum(df2)
    df2["RSI"] = compute_rsi(df2)
    macd_df = compute_macd(df2)
    df2 = pd.concat([df2, macd_df], axis=1)
    df2["Volume_Surge"] = compute_volume_surge(df2)
    df2["Shock"] = detect_shock(df2)
    return df2.dropna()


def normalize(series: pd.Series) -> pd.Series:
    """Min-max normalize a pandas Series to [0,1] safely."""
    minv = series.min()
    maxv = series.max()
    if pd.isna(minv) or pd.isna(maxv) or maxv == minv:
        return pd.Series(0.0, index=series.index)
    return (series - minv) / (maxv - minv)


def build_risk_index(df: pd.DataFrame,
                     w_vol: float = 0.4,
                     w_mom: float = 0.4,
                     w_vol_surge: float = 0.2) -> pd.Series:
    """
    Build a risk index from Volatility, Momentum and Volume_Surge.
    Weights sum should be 1.0 by default.
    """
    v = normalize(df["Volatility"])
    m = normalize(df["Momentum"].abs())  # use magnitude of momentum
    vs = normalize(df["Volume_Surge"].abs())

    risk = w_vol * v + w_mom * m + w_vol_surge * vs
    # Clip to [0,1]
    return risk.clip(0, 1)


import pandas as pd
import numpy as np

def process_ticker(ticker, start_date, end_date):
    """
    Fetch OHLCV data and compute all indicators including:
    Volatility, ATR, Momentum, RSI, MACD, Volume Surge, Shock Flags, Risk Index
    """
    df = fetch_price_data(ticker, start_date, end_date)
    
    # Daily returns
    df["Return"] = df["Close"].pct_change()
    
    # Rolling 30-day volatility
    df["Volatility"] = df["Return"].rolling(30).std()
    
    # ATR (Average True Range)
    df["H-L"] = df["High"] - df["Low"]
    df["H-Cp"] = (df["High"] - df["Close"].shift(1)).abs()
    df["L-Cp"] = (df["Low"] - df["Close"].shift(1)).abs()
    df["TR"] = df[["H-L", "H-Cp", "L-Cp"]].max(axis=1)
    df["ATR"] = df["TR"].rolling(14).mean()
    
    # Momentum (30-day)
    df["Momentum"] = df["Close"].pct_change(30)
    
    # RSI (14-day)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    
    # Volume surge (7-day rolling % change)
    df["Volume_Surge"] = df["Volume"].pct_change().rolling(7).mean()
    
    # Shock flag (>10% in 5 days)
    df["Shock"] = (df["Close"].pct_change(5) > 0.1).astype(int)
    
    # Risk Index (0-1 scaled)
    df["Vol_Scaled"] = df["Volatility"] / df["Volatility"].max()
    df["Mom_Scaled"] = df["Momentum"] / df["Momentum"].abs().max()
    df["Vol_Surge_Scaled"] = df["Volume_Surge"] / df["Volume_Surge"].max()
    df["Risk_Index"] = df["Vol_Scaled"]*0.4 + df["Mom_Scaled"]*0.4 + df["Vol_Surge_Scaled"]*0.2
    
    # Clean up intermediates
    df.drop(columns=["H-L", "H-Cp", "L-Cp", "TR", "Vol_Scaled", "Mom_Scaled", "Vol_Surge_Scaled"], inplace=True)
    
    return df

