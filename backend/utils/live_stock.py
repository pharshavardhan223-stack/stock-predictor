"""
backend/utils/live_stock.py

Utility for fetching live stock data and saving to CSV.
Used by the /live route for CSV download/preview functionality.
"""

import os
import datetime
import yfinance as yf
import pandas as pd


def get_live_stock(symbol: str) -> str | None:
    """
    Fetch today's 1-minute interval data for a symbol,
    save to a CSV file, and return the file path.

    Returns None if the symbol is invalid or data unavailable.
    """
    try:
        ticker = yf.Ticker(symbol.upper())
        data   = ticker.history(period="1d", interval="1m")

        if data is None or data.empty:
            return None

        os.makedirs("data", exist_ok=True)
        filename  = f"live_{symbol.upper()}_{datetime.date.today()}.csv"
        file_path = os.path.join("data", filename)
        data.to_csv(file_path)
        return file_path

    except Exception as e:
        print(f"get_live_stock error ({symbol}): {e}")
        return None