import yfinance as yf
import pandas as pd
import os


def get_live_stock(symbol, days=60):

    try:
        stock = yf.Ticker(symbol)

        data = stock.history(period=f"{days}d")

        if data.empty:
            return None


        df = data.reset_index()

        df = df[["Date", "Close"]]

        df.columns = ["date", "price"]


        os.makedirs("uploads", exist_ok=True)

        file_path = f"uploads/{symbol}_live.csv"

        df.to_csv(file_path, index=False)

        return file_path


    except Exception as e:
        print("Stock API Error:", e)
        return None
