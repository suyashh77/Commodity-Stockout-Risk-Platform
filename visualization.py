# visualization.py
"""
Visualization functions that return matplotlib figures so Streamlit can show them.
"""

from typing import List
import matplotlib.pyplot as plt
import pandas as pd


def plot_price(df: pd.DataFrame, column: str = "Adj Close") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.index, df[column], label=column)
    ax.set_title("Price")
    ax.set_ylabel("Price")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_volatility(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(df.index, df["Volatility"], label="Rolling Volatility (30d)")
    ax.set_title("Rolling Volatility")
    ax.set_ylabel("Volatility")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_atr(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(df.index, df["ATR"], label="ATR (14d)")
    ax.set_title("Average True Range (ATR)")
    ax.set_ylabel("ATR")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_momentum_and_rsi(df: pd.DataFrame) -> List[plt.Figure]:
    fig1, ax1 = plt.subplots(figsize=(12, 3))
    ax1.plot(df.index, df["Momentum"], label="Momentum (30d)")
    ax1.set_title("Momentum (30d)")
    ax1.grid(True)
    ax1.legend()
    plt.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(12, 3))
    ax2.plot(df.index, df["RSI"], label="RSI")
    ax2.axhline(70, color="red", linestyle="--", linewidth=0.6)
    ax2.axhline(30, color="green", linestyle="--", linewidth=0.6)
    ax2.set_title("RSI")
    ax2.grid(True)
    ax2.legend()
    plt.tight_layout()

    return [fig1, fig2]


def plot_macd(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(df.index, df["MACD"], label="MACD")
    ax.plot(df.index, df["MACD_Signal"], label="MACD Signal")
    ax.set_title("MACD")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_volume_surge(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(df.index, df["Volume_Surge"], label="Volume Surge (rolling mean pct change)")
    ax.set_title("Volume Surge")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_risk_index(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(df.index, df["Risk_Index"], label="Risk Index", color="purple")
    ax.set_title("Risk Index (0 - 1)")
    ax.set_ylabel("Risk")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_alerts_on_price(df: pd.DataFrame) -> plt.Figure:
    """Plot price and mark Shock events and high Risk_Index days."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.index, df["Adj Close"], label="Adj Close")
    shocks = df.index[df["Shock"] == 1]
    ax.scatter(shocks, df.loc[shocks, "Adj Close"], color="red", label="Shock", zorder=5)
    high_risk = df.index[df["Risk_Index"] > 0.75]
    ax.scatter(high_risk, df.loc[high_risk, "Adj Close"], color="orange", label="High Risk", marker="x", zorder=5)
    ax.set_title("Price with Shock & High Risk Annotations")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    return fig
