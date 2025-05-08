import streamlit as st
import pandas as pd
import numpy as np
import subprocess
import os
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
import yfinance as yf

st.set_page_config(page_title="📈 Stock Prediction App", layout="wide")
st.title("🔮 AI-Based Stock Price Prediction")

# --- Sidebar ---
with st.sidebar:
    st.header("🔧 Search & Settings")
    stock = st.text_input("Enter Stock Symbol (e.g. AAPL, TSLA, INFY.NS)", "AAPL").upper()
    window = st.slider("Training Window Size (days)", 30, 100, 60)
    forecast_days = st.slider("Forecast Days", 10, 60, 30)
    
    if st.button("🔁 Train & Predict"):
        with st.spinner("🔄 Downloading data, training model, and generating predictions..."):
            try:
                predictions_file = f"predictions_{stock}.csv"

                if os.path.exists(predictions_file):
                    os.remove(predictions_file)

                subprocess.run(["python", "machinelearning.py", stock, str(window)], check=True)
                st.success("✅ Model trained and predictions generated!")
            except Exception as e:
                st.error(f"❌ Error: {e}")

# --- Show Predictions ---
try:
    predictions_file = f"predictions_{stock}.csv"

    if not os.path.exists(predictions_file):
        st.warning(f"⚠ No predictions found for {stock}. Please click 'Train & Predict' to generate predictions.")
        st.stop()

    df = pd.read_csv(predictions_file, parse_dates=["Date"])

    st.subheader("📊 Historical vs Predicted Prices")
    st.line_chart(df.set_index("Date")[["Actual Price", "Predicted Price"]])

    st.subheader("📋 Latest Predictions (Last 10 Days)")
    latest_10 = df.tail(10)
    st.dataframe(latest_10, use_container_width=True)

    # --- Predict Next N Days ---
    st.subheader(f"📌 Next {forecast_days}-Day Price Forecast")
    data = yf.download(stock, period=f"{window + forecast_days}d")[["Close"]].dropna()
    recent = data["Close"].values[-window:]

    min_ = np.load("scaler_min.npy")[0]
    max_ = np.load("scaler_max.npy")[0]

    model = load_model("keras_model.h5", compile=False)

    forecast = []
    input_seq = recent.copy()
    for _ in range(forecast_days):
        scaled_input = (input_seq[-window:] - min_) / (max_ - min_)
        scaled_input = scaled_input.reshape(1, window, 1)
        pred_scaled = model.predict(scaled_input)[0][0]
        next_price = pred_scaled * (max_ - min_) + min_
        forecast.append(next_price)
        input_seq = np.append(input_seq, next_price)

    future_dates = [datetime.today() + timedelta(days=i + 1) for i in range(forecast_days)]
    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted Price": forecast
    })

    st.dataframe(forecast_df, use_container_width=True)
    st.line_chart(forecast_df.set_index("Date")["Predicted Price"])

    # --- Forecast Summary ---
    st.subheader("📈 Forecast Summary")
    predicted_high = latest_10["Predicted Price"].max()
    predicted_low = latest_10["Predicted Price"].min()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🔺 Forecasted High", f"₹{predicted_high:.2f}")
    with col2:
        st.metric("🔻 Forecasted Low", f"₹{predicted_low:.2f}")

    st.subheader("📉 Risk Level Analysis")
    diff = predicted_high - predicted_low
    if diff > 15:
        st.error("🚨 High Risk Detected: Significant price fluctuations expected.")
    elif diff > 7:
        st.warning("⚠ Medium Risk: Moderate fluctuation.")
    else:
        st.success("✅ Low Risk: Price range appears stable.")

except Exception as e:
    st.warning(f"⚠ Error loading predictions: {e}")
