import sys
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense  # Added Dense import
import os

def train_model(stock_symbol="AAPL", window_size=60):
    # Fetch updated data
    df = yf.download(stock_symbol, period="5y")
    df = df[["Close"]].dropna()

    # Scale data
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(window_size, len(scaled)):
        X.append(scaled[i - window_size:i])
        y.append(scaled[i])

    X, y = np.array(X), np.array(y)

    # Model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Train model
    model.fit(X, y, epochs=5, batch_size=32)

    # Save model and scaler
    model.save("keras_model.h5")  # Saved with a generic name for app compatibility
    np.save("scaler_min.npy", [scaler.data_min_[0]])
    np.save("scaler_max.npy", [scaler.data_max_[0]])

    # Predict all points
    predicted_scaled = model.predict(X)
    predicted = predicted_scaled * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]
    actual = y * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]

    # Create prediction DataFrame with correct dates
    prediction_dates = df.index[window_size:]
    pred_df = pd.DataFrame({
        "Date": prediction_dates,
        "Actual Price": actual.flatten(),
        "Predicted Price": predicted.flatten()
    })

    # Save to stock-specific CSV
    output_file = f"predictions_{stock_symbol.upper()}.csv"
    pred_df.to_csv(output_file, index=False)
    print(f"Saved predictions to {output_file}")

# Run from command line
if __name__ == "__main__":
    stock = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    window = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    train_model(stock, window)
