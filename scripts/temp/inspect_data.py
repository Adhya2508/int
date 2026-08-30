import pandas as pd
import sys

file_path = "e:/intain/2025Q1.csv"
try:
    # Read just a small chunk to inspect
    df = pd.read_csv(file_path, sep="|", header=None, nrows=100)
    print("Shape:", df.shape)
    print("Num columns:", len(df.columns))
    print(df.head(5).to_string())
except Exception as e:
    print(f"Error: {e}")
