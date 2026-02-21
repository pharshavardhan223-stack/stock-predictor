import pandas as pd
import io

def load_csv(filepath, max_rows=10000):

    # Read file as binary
    with open(filepath, "rb") as f:
        raw = f.read()

    # Try decoding with common encodings
    encodings = ["utf-8", "utf-16", "latin1", "cp1252"]

    text = None

    for enc in encodings:
        try:
            text = raw.decode(enc)
            break
        except:
            continue

    if text is None:
        raise ValueError("File is not a valid CSV text file")

    # Read from decoded text
    df = pd.read_csv(io.StringIO(text))

    if len(df) > max_rows:
        df = df.head(max_rows)

    return df


def clean_data(df):

    df = df.dropna(how="all")

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())

    return df


def preview_table(df, rows=5):

    return df.head(rows).to_html(classes="table table-bordered")
