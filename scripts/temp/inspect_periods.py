import pandas as pd
import os

raw_dir = "e:/intain/data"
train_raw = pd.read_csv(os.path.join(raw_dir, "loan_monthly_performance_train.csv"))
test_raw = pd.read_csv(os.path.join(raw_dir, "loan_monthly_performance_test.csv"))

print("Train raw reporting_period unique values:")
print(sorted(train_raw['reporting_period'].unique()))

print("\nTest raw reporting_period unique values:")
print(sorted(test_raw['reporting_period'].unique()))
