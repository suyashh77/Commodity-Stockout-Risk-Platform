
import os
from dotenv import load_dotenv
import openai
import pandas as pd

# Load your API key from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def generate_commodity_summary(df: pd.DataFrame, commodity: str, model_label: int, model_proba: float) -> str:
    """
    Generates a 4-line summary explaining potential stockout risk for a commodity.
    Uses OpenAI GPT-4 via the new API interface.
    """
    latest = df.iloc[-1]
    indicators = {
        "ATR": latest.get("ATR", None),
        "Volatility": latest.get("Volatility", None),
        "Momentum": latest.get("Momentum", None),
        "RSI": latest.get("RSI", None),
        "MACD": latest.get("MACD", None),
        "Volume_Surge": latest.get("Volume_Surge", None),
        "Risk_Index": latest.get("Risk_Index", None)
    }

    prompt = f"""
    You are an expert supply chain analyst. 
    Here are the latest indicators for {commodity}:

    {indicators}

    The ML model predicts shock = {model_label} (probability {model_proba:.2%}).

    Write a concise 4-line summary explaining whether this commodity is at risk of a potential stockout in the near future, referencing the indicators. Keep it easy to understand.
    """

    # New API interface
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )

    summary_text = response.choices[0].message.content.strip()
    return summary_text
