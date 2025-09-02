# model_training.py
"""
Simple ML training utilities for predicting Shock events from features.
This module contains pure functions: no top-level execution.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd


def prepare_ml_data(df: pd.DataFrame, label_col: str = "Shock", feature_cols: list = None):
    """
    Returns X, y for modeling.
    If feature_cols is None, use a sensible default set.
    """
    if feature_cols is None:
        feature_cols = ["Volatility", "ATR", "Momentum", "RSI", "MACD", "MACD_Signal", "Volume_Surge", "Risk_Index"]
    X = df[feature_cols].dropna()
    y = df.loc[X.index, label_col]
    return X, y


def train_model(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """
    Train a RandomForest classifier and return the trained model and test accuracy.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    return model, acc


def predict_today(model, df: pd.DataFrame, feature_cols: list = None):
    """
    Predict the probability of a shock for the latest available row.
    Returns: (pred_label, pred_proba)
    """
    if feature_cols is None:
        feature_cols = ["Volatility", "ATR", "Momentum", "RSI", "MACD", "MACD_Signal", "Volume_Surge", "Risk_Index"]
    latest = df[feature_cols].tail(1)
    proba = model.predict_proba(latest)[0][1]  # probability of class 1
    label = model.predict(latest)[0]
    return label, float(proba)
