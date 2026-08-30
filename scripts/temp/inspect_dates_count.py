import pandas as pd
import os

train = pd.read_csv("e:/intain/data_cleaned/train_modeling_ready.csv")
train['reporting_date'] = pd.to_datetime(train['reporting_date'])

print("Total train rows:", len(train))
print("Rows where reporting_date >= '2025-12-01':", len(train[train['reporting_date'] >= '2025-12-01']))
print("Rows where reporting_date < '2025-12-01':", len(train[train['reporting_date'] < '2025-12-01']))
