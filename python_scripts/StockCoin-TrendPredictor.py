#!/usr/bin/env python3
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys
import json

def get_data(ticker, period='1y'):
    """
    Download historical data for the given ticker using yfinance.
    """
    data = yf.download(ticker, period=period)
    return data

def create_features(data, window=5):
    """
    Create a DataFrame with features based on a sliding window.
    For example, if window=5, we use the closing prices of the previous 5 days
    to predict the current day's closing price.
    """
    df = data[['Close']].copy()
    # Create lag features
    for i in range(1, window + 1):
        df[f'lag_{i}'] = df['Close'].shift(i)
    # Drop any rows with NaN values (due to shifting)
    df.dropna(inplace=True)
    return df

def train_model(df):
    """
    Train a linear regression model using the lag features.
    Returns the trained model and the test data for evaluation.
    """
    # Our features are lag_1, lag_2, ..., lag_window; the target is today's closing price.
    X = df.drop(columns=['Close'])
    y = df['Close']
    
    # Split the data into training and testing sets
    # Here we avoid shuffling to preserve time series order.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    score = model.score(X_test, y_test)
    print(f"Model R² score on test data: {score:.4f}")
    return model, X_test, y_test

def predict_future(model, data, window=5, days=10):
    """
    Predict future closing prices for a specified number of days.
    Uses the last available window of actual data and then iteratively predicts future days.
    """
    predictions = []
    # Start with the last available 'window' closing prices
    last_window = data['Close'][-window:].values.tolist()
    
    for _ in range(days):
        # Prepare features from the most recent window
        features = np.array(last_window[-window:]).reshape(1, -1)
        pred = model.predict(features)[0]
        predictions.append(pred)
        # Append the prediction so it becomes part of the window for the next prediction
        last_window.append(pred)
    
    return predictions

def plot_predictions(data, predictions, days=10):
    """
    Plot historical closing prices along with the predicted future prices.
    """
    plt.figure(figsize=(12, 6))
    
    # Plot historical data
    plt.plot(data.index, data['Close'], label="Historical Close")
    
    # Create future dates for the predictions
    last_date = data.index[-1]
    # Generate a date range including the last_date and future days
    full_dates = pd.date_range(start=last_date, periods=days+1)
    # Exclude the first date (last_date) to mimic closed='right'
    future_dates = full_dates[1:]
    
    plt.plot(future_dates, predictions, label="Predicted Future Close", linestyle='--', marker='o')
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Stock/Cryptocurrency Price Prediction")
    
    # Set x-axis limits to start
    plt.xlim(pd.Timestamp('2024-10-01'), future_dates[-1])
    
    plt.legend()
    plt.show()

def main(ticker):
    """
    Run the prediction process using the provided ticker symbol.
    Downloads data, creates features, trains a model, and predicts future prices.
    Returns the result as a dictionary.
    """
    data = get_data(ticker, period='1y')
    if data.empty:
        error_msg = f"No data found for {ticker}. Please check the ticker symbol and try again."
        print(json.dumps({"error": error_msg}))
        sys.exit(1)
    
    # Define the number of lag days to use as features (you can experiment with this)
    window = 5
    df_features = create_features(data, window=window)
    
    # Train the model and evaluate its performance
    model, X_test, y_test = train_model(df_features)
    
    # Predict future prices for the next N days
    future_days = 10
    predictions = predict_future(model, data, window=window, days=future_days)
    print(f"Predicted closing prices for the next {future_days} days: {predictions}")
    
    # Optionally, you could comment out the plot if running in an API context
    # plot_predictions(data, predictions, days=future_days)
    
    result = {
        "ticker": ticker,
        "future_days": future_days,
        "predictions": predictions
    }
    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No ticker provided. Usage: python3 StockCoin-TrendPredictor.py <TICKER>"}))
        sys.exit(1)
    ticker = sys.argv[1]
    result = main(ticker)
    print(json.dumps(result))
