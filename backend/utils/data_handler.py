# backend/utils/data_handler.py

import pandas as pd
import numpy as np

def load_csv(filepath, max_rows=None):
    """Load CSV file with proper error handling"""
    try:
        df = pd.read_csv(filepath)
        
        # Convert date column if exists
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        
        # Ensure all data is numeric
        df = df.apply(pd.to_numeric, errors='coerce')
        
        if max_rows and len(df) > max_rows:
            df = df.tail(max_rows)
            
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return pd.DataFrame()

def clean_data(df):
    """Clean dataframe - remove NaN and infinite values"""
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    return df

def preview_table(df, limit=10):
    """Generate HTML preview of dataframe"""
    return df.head(limit).to_html(classes="table table-striped", border=0)