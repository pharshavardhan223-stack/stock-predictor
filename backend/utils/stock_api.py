import yfinance as yf
import pandas as pd
import os
from datetime import datetime


def fetch_stock_data(symbol, period="1y", interval="1d"):
    """
    Fetch stock data from Yahoo Finance and save as CSV

    Args:
        symbol (str): Stock symbol (AAPL, TSLA, RELIANCE.NS)
        period (str): Data period (1mo, 3mo, 6mo, 1y, 5y)
        interval (str): Data interval (1d, 1wk, 1mo)

    Returns:
        str: Path of saved CSV file
    """

    try:
        # Download stock data
        stock = yf.Ticker(symbol)

        df = stock.history(period=period, interval=interval)

        # Check if data exists
        if df.empty:
            return None


        # Reset index (Date becomes column)
        df.reset_index(inplace=True)


        # Keep useful columns only
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]


        # Create uploads folder if not exists
        base_dir = os.path.abspath(os.path.dirname(__file__))
        upload_dir = os.path.join(base_dir, "..", "uploads")

        os.makedirs(upload_dir, exist_ok=True)


        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"{symbol}_{timestamp}.csv"

        filepath = os.path.join(upload_dir, filename)


        # Save as CSV
        df.to_csv(filepath, index=False)


        return filepath


    except Exception as e:
        print("Stock Fetch Error:", e)
        return None
import yfinance as yf


def get_live_price(symbol):

    stock = yf.Ticker(symbol)

    data = stock.history(period="1d")

    if data.empty:
        return None

    price = round(data["Close"].iloc[-1], 2)

    return {
        "symbol": symbol,
        "price": price
    }
