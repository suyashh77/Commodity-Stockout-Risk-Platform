# Commodity Stockout Risk Platform

This project is a **Streamlit-based web application** that monitors potential **stockout risks for commodities** using market signals from financial data. The app uses **prices, volume, and technical indicators** as proxies to detect supply chain stress, and leverages machine learning to provide short-term early-warning signals.

---

##  Key Features

1. **Market Data Fetching**  
   - Downloads OHLCV (Open, High, Low, Close, Volume) data from Yahoo Finance using `yfinance`.  
   - Supports commodities and ETFs like Lithium (LIT), Gold (GC=F), Silver (SI=F), Steel (SLX), Copper (HG=F), and Aluminium (ALI=F).  

2. **Technical Indicators & Risk Signals**  
   - **ATR (Average True Range):** Measures daily price movements → higher ATR = more instability.  
   - **Rolling Volatility:** Standard deviation of returns → captures turbulence.  
   - **Momentum:** Tracks price trends → sustained increases indicate possible supply shortage.  
   - **RSI (Relative Strength Index):** Shows overbought (>70) or oversold (<30) conditions.  
   - **MACD (Moving Average Convergence Divergence):** Highlights trend shifts.  
   - **Volume Surge:** Detects sudden spikes in trading volume → potential market reaction to supply/demand changes.  
   - **Shock Flags:** Detects unusual rapid price moves (e.g., >10% in 5 days).  
   - **Stockout Risk Index (0–1):** Combines scaled indicators into a single score where 1 = high risk of supply stress.

3. **Machine Learning Proxy Model**  
   - Uses **Random Forest Classifier** from `scikit-learn` to predict the likelihood of a shock event based on past indicator values.  
   - Model accuracy is computed using test data and probability of a shock event is displayed for the latest date.  
   - Provides an **early-warning signal**, not an exact stockout forecast.

4. **Interactive Streamlit App**  
   - **Landing Page:** Users select a commodity and click “Go” to fetch data and compute indicators.  
   - **Visualizations:** Displays graphs for OHLC prices, volatility, ATR, Momentum, RSI, MACD, Volume Surge, Risk Index, and alerts.  
   - **GPT-4 Summary:** Generates a 4-line human-readable summary explaining the stockout risk for the commodity using OpenAI GPT-4 API.

5. **Stable & Robust Processing**  
   - Handles missing data for ETFs/futures like Aluminium.  
   - Computes all indicators safely with forward-filling and rolling windows.  
   - API key for OpenAI loaded securely from `.env`.

---

## 🔹 Project Structure
commodity_platform/
│
├─ data_processing.py # Fetches commodity data, computes returns, volatility, ATR, momentum, RSI, MACD, volume surge, shock flags, risk index
├─ visualization.py # Functions to plot price, volatility, ATR, momentum, RSI, MACD, volume surge, risk index, and alerts
├─ model_training.py # Prepares ML features, trains Random Forest, predicts short-term shock events
├─ utils.py # Helper functions including GPT summary generator
├─ streamlit_app.py # Main interactive app integrating all modules
├─ requirements.txt # Python dependencies
└─ README.md # Project documentation

---

##  How It Works (Collapsible Explanation in the App)

1. **Fetch Market Data:**  
   Downloads OHLCV data from Yahoo Finance for the chosen commodity.

2. **Compute Early Warning Indicators:**  
   Calculates ATR, volatility, momentum, RSI, MACD, volume surges, and shock flags.

3. **Combine into Risk Index:**  
   Scales each indicator (0–1) and creates a **Stockout Risk Index** representing supply stress.

4. **Train Machine Learning Model:**  
   Random Forest Classifier predicts short-term market proxy shock events based on indicators.  

5. **Generate GPT Summary:**  
   Sends latest indicator values and model prediction to OpenAI GPT-4 and produces a concise 4-line summary explaining stockout risk.

