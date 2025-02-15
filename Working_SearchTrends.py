import pandas as pd
import numpy as np
import yfinance as yf
from pytrends.request import TrendReq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ============================
# CONFIGURATION
# ============================
TICKER = "AAPL"                  # Stock ticker
START_DATE = "2025-01-01"        # Start date for both stock and trends data
END_DATE = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
KEYWORD = "Apple"                # Keyword to query on Google Trends

# ============================
# FUNCTIONS TO RETRIEVE DATA
# ============================
def get_stock_data(ticker, start_date, end_date):
    """
    Downloads historical stock data using yfinance.
    """
    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty:
        raise ValueError("No stock data downloaded. Check the ticker and date range.")
    
    # Reset the index so the date becomes a column
    df.reset_index(inplace=True)
    
    # Flatten multi-level columns if they exist
    if isinstance(df.columns, pd.MultiIndex):
        # For a single ticker, we can take only the first level (field name)
        df.columns = df.columns.get_level_values(0)
    
    # Ensure 'Date' is in datetime format and normalize to remove any time component
    df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
    return df

def get_google_trends_data(keyword, start_date, end_date):
    """
    Retrieves daily Google Trends data for the specified keyword and date range.
    """
    # Build a timeframe string in the format "YYYY-MM-DD YYYY-MM-DD"
    timeframe = f"{start_date} {end_date}"
    
    # Initialize pytrends
    pytrends = TrendReq(hl='en-US', tz=360)
    pytrends.build_payload([keyword], cat=0, timeframe=timeframe, geo='', gprop='')
    
    # Retrieve interest over time
    trends_df = pytrends.interest_over_time()
    if trends_df.empty:
        raise ValueError("No trends data returned. Check your keyword and date range.")
    
    # Remove the "isPartial" column if it exists
    if 'isPartial' in trends_df.columns:
        trends_df = trends_df.drop(columns=['isPartial'])
    
    # Reset index to convert the date index into a column, and rename columns appropriately
    trends_df.reset_index(inplace=True)
    
    # If the date column is not named 'Date', rename it (sometimes it's 'date')
    if 'date' in trends_df.columns:
        trends_df.rename(columns={'date': 'Date'}, inplace=True)
    
    # Rename the keyword column to 'buying_trend'
    trends_df.rename(columns={keyword: 'buying_trend'}, inplace=True)
    
    # Ensure the 'Date' column is datetime and normalized
    trends_df['Date'] = pd.to_datetime(trends_df['Date']).dt.normalize()
    return trends_df

# ============================
# MAIN PIPELINE
# ============================
def main():
    # 1. Get historical stock data
    print("Downloading stock data...")
    stock_data = get_stock_data(TICKER, START_DATE, END_DATE)
    print(stock_data.head())
    
    # 2. Get Google Trends data for the given keyword
    print("\nDownloading Google Trends data...")
    trends_data = get_google_trends_data(KEYWORD, START_DATE, END_DATE)
    print(trends_data.head())
    
    # Debug: Print column names and types for both DataFrames
    print("\nStock Data columns and types:")
    print(stock_data.dtypes)
    print("\nTrends Data columns and types:")
    print(trends_data.dtypes)
    
    # 3. Merge the datasets on Date
    try:
        merged_data = pd.merge(stock_data, trends_data, on="Date", how="left")
    except Exception as e:
        print("Merge error:", e)
        return

    # Fill missing trend values (if any) by forward/backward filling
    merged_data['buying_trend'] = merged_data['buying_trend'].fillna(method='ffill').fillna(method='bfill')
    
    print("\nMerged Data Sample:")
    print(merged_data.head())
    
    # 4. Create target variable: 1 if next day's closing price is higher, else 0.
    merged_data['target'] = (merged_data['Close'].shift(-1) > merged_data['Close']).astype(int)
    merged_data = merged_data.dropna().copy()
    
    # 5. Define feature columns and prepare training data.
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'buying_trend']
    X = merged_data[feature_cols]
    y = merged_data['target']
    
    # 6. Split data into training and testing sets (time-ordered split)
    split_index = int(0.8 * len(merged_data))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    # 7. Train a Logistic Regression classifier.
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # 8. Evaluate the model.
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    print(f"\nModel Accuracy on Test Set: {accuracy:.2f}")
    
    # 9. Predict the next day's movement using the latest available data.
    latest_features = merged_data.iloc[-1][feature_cols].values.reshape(1, -1)
    prediction = model.predict(latest_features)
    prediction_text = "Up" if prediction[0] == 1 else "Down"
    print(f"\nPrediction for the next day's movement: {prediction_text} (1 means Up, 0 means Down)")
    
    # 10. (Optional) Plot the Google Trends data.
    plt.figure(figsize=(10, 4))
    plt.plot(trends_data['Date'], trends_data['buying_trend'], marker='o', linestyle='-')
    plt.title(f'Google Trends Buying Trend for "{KEYWORD}"')
    plt.xlabel("Date")
    plt.ylabel("Search Interest")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    main()
