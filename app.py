import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import torch
import joblib
import yfinance as yf
from datetime import datetime, timedelta
from config import DEVICE, feature_cols, SEQ_LEN, H

# Charger modèle et scaler
from model import LSTMDirect, predict_future  # suppose qu’on a mis notre classe dans model.py

# Recréer le modèle et charger les poids
model = LSTMDirect(input_size=len(feature_cols), horizon=H)
model.load_state_dict(torch.load("btc_eur_lstm_direct.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Recharger les scalers
scaler_X = joblib.load("scaler_X.save")
scaler_y = joblib.load("scaler_y.save")

# UI
st.title("Prévisions financières avec LSTM")
ticker = st.text_input("Choisir un ticker (ex: BTC-USD, AAPL)", "BTC-USD")
future_days = st.slider("Nombre de jours à prévoir", 1, 60, 30)

if st.button("Lancer prévision"):
    df = yf.download(ticker, start="2018-01-01")
    df = df[['Close']].dropna()

    # Ajouter les mêmes features que pour l'entraînement
    df["ma_10"] = df["Close"].rolling(10).mean()
    df["ma_30"] = df["Close"].rolling(30).mean()
    df["vol_10"] = df["Close"].pct_change().rolling(10).std()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi_14"] = 100 - 100/(1+rs)
    df = df.dropna()

    # Normalisation
    X_scaled = scaler_X.transform(df[feature_cols])
    last_seq = X_scaled[-SEQ_LEN:]

    
    preds = predict_future(model, last_seq, future_days, scaler_y, H=H, device=DEVICE)
    
    future_index = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=future_days)
    
    # Visualisation
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(df.index, df["Close"], label="Historique")
    ax.plot(future_index, preds, label="Prévisions")
    ax.legend()
    st.pyplot(fig)
