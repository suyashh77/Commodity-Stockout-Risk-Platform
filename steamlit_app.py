# streamlit_app.py
"""
Streamlit app that ties everything together.
Run: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
from data_processing import COMMODITY_MAP, process_ticker
from visualization import (
    plot_price, plot_volatility, plot_atr, plot_momentum_and_rsi,
    plot_macd, plot_volume_surge, plot_risk_index, plot_alerts_on_price
)
from model_training import prepare_ml_data, train_model, predict_today
import matplotlib.pyplot as plt

st.set_page_config(page_title="Commodity Stockout Risk Platform", layout="wide")

# --- Landing UI ---
st.title("Commodity Stockout Risk Platform")

st.markdown("This app uses **market signals (prices, volume, technical indicators)** as a proxy to monitor **supply risk** for commodities.")

# --- Collapsible sections ---
with st.expander("1️. Pull Market Data"):
    st.markdown("""
    - Download daily **OHLCV** data (Open, High, Low, Close, Volume) from Yahoo Finance.  
    - These numbers describe how the price moved and how many contracts were traded.
    """)

with st.expander("2️. Compute Early Warning Indicators"):
    st.markdown("""
    - **ATR (Average True Range):** Captures daily price movement → higher = more instability.  
    - **Rolling Volatility:** Standard deviation of returns → measures price turbulence.  
    - **Momentum:** Price change trend (up = possible shortage, down = easing supply).  
    - **RSI (Relative Strength Index):** 0–100 scale of overbought (>70) vs oversold (<30).  
    - **MACD (Moving Average Convergence Divergence):** Spots shifts in market direction.  
    - **Volume Surge:** Sudden spikes in trading → traders expect supply/demand shocks.  
    - **Shock Flags:** Detects unusual price jumps (e.g. >10% in 5 days).  
    """)

with st.expander("3️. Combine into Stockout Risk Index"):
    st.markdown("""
    - Each signal is scaled 0–1.  
    - Weighted into a single **risk index**:  
        - `0 = stable market (low risk)`  
        - `1 = unstable market (high stress)`
    """)

with st.expander("4️. Train the Model"):
    st.markdown("""
    - We use a Random Forest Classifier from scikit-learn to predict whether a large sudden price move (‘Shock Event’) is likely, based on past market indicators like ATR, volatility, momentum, RSI, MACD, and volume surges. 
    - This serves as a market-based proxy for short-term supply stress, not actual inventory shortages
    """)

with st.expander("5️. Limitations"):
    st.markdown("""
    - Uses only **financial data** (no direct inventory or demand inputs).  
    - Best as a **proxy signal** for potential shortages.  
    """)

st.markdown("---")


# Sidebar controls
commodity_choice = st.sidebar.selectbox("Select Commodity", list(COMMODITY_MAP.keys()))
start_date = st.sidebar.date_input("Start date", value=pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End date (last inclusive)", value=pd.to_datetime("2025-09-01"))

if st.sidebar.button("Go"):
    ticker = COMMODITY_MAP[commodity_choice]
    st.info(f"Fetching {commodity_choice} ({ticker}) data and computing indicators...")

    # --- Processing ---
    df = process_ticker(ticker, start_date=str(start_date), end_date=str(end_date))

    # --- Top-level summary ---
    st.header(f"{commodity_choice} — Overview")
    st.write(df.tail(5))

    # --- Visualizations ---
    st.subheader("Price & Alerts")
    fig_price_alerts = plot_alerts_on_price(df)
    st.pyplot(fig_price_alerts)

    st.subheader("Price")
    st.pyplot(plot_price(df, column="Adj Close"))

    st.subheader("Volatility")
    st.pyplot(plot_volatility(df))

    st.subheader("ATR")
    st.pyplot(plot_atr(df))

    st.subheader("Momentum & RSI")
    figs = plot_momentum_and_rsi(df)
    for f in figs:
        st.pyplot(f)

    st.subheader("MACD")
    st.pyplot(plot_macd(df))

    st.subheader("Volume Surge")
    st.pyplot(plot_volume_surge(df))

    st.subheader("Risk Index")
    st.pyplot(plot_risk_index(df))

    # --- Modeling ---
    st.subheader("Modeling: Predict Shock (market proxy event)")
    X, y = prepare_ml_data(df)
    if len(y.unique()) <= 1:
        st.warning("Not enough variation in target 'Shock' to train a classifier. Try a different ticker or wider date range.")
    else:
        model, acc = train_model(X, y)
        label, proba = predict_today(model, df)
        st.write(f"Model test accuracy: **{acc:.2%}**")
        st.write(f"Predicted shock label for latest date: **{label}** (probability = {proba:.2%})")
        if proba > 0.5:
            st.error("Market indicates increased short-term supply risk (proxy).")
        else:
            st.success(" Low short-term market-signal supply risk (proxy).")

        # --- GPT Summary ---
        from utils import generate_commodity_summary
        summary_text = generate_commodity_summary(df, commodity_choice, label, proba)
        st.subheader(" Stockout Risk Summary")
        st.write(summary_text)
        st.write("This summary was generated using OpenAI API - GPT 4o mini")
        