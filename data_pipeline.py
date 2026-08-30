import pandas as pd
import numpy as np
import os

def process_data():
    input_file = "e:/intain/2025Q1.csv"
    print("Reading data...")
    # Read the first 500,000 rows for manageable processing
    df = pd.read_csv(input_file, sep="|", header=None, nrows=500000, low_memory=False)
    
    # Standard mapping for Fannie Mae Single-Family Loan Performance Data
    columns = [
        "reference_pool_id", "loan_id", "reporting_period", "channel", "seller", "servicer", 
        "master_servicer", "orig_interest_rate", "current_interest_rate", "orig_upb", 
        "upb_at_issuance", "current_upb", "orig_loan_term", "orig_date", "first_payment_date", 
        "loan_age", "remaining_months", "remaining_months_to_maturity", "maturity_date", 
        "ltv", "cltv", "num_borrowers", "dti", "credit_score", "c_credit_score", "first_time_buyer",
        "loan_purpose", "property_type", "num_units", "occupancy_status", "state", "zip", 
        "mortgage_insurance_pct", "amortization_type", "prepayment_penalty", "interest_only_indicator", 
        "interest_only_first_payment_date", "months_to_amortization", "delinquency_status", "loan_payment_history"
    ]
    
    # Pad columns if dataset has more
    if len(df.columns) > len(columns):
        for i in range(len(columns), len(df.columns)):
            columns.append(f"col_{i}")
            
    df.columns = columns[:len(df.columns)]
    
    print("Extracting Static Attributes...")
    static_cols = ['loan_id', 'orig_upb', 'credit_score', 'ltv', 'dti', 'state', 'loan_purpose', 'property_type', 'orig_date']
    static_df = df[static_cols].drop_duplicates(subset=['loan_id'], keep='first')
    static_df.rename(columns={'orig_date': 'vintage'}, inplace=True)
    static_df.to_csv("loan_static_attributes.csv", index=False)
    
    print("Extracting Monthly Performance...")
    monthly_cols = ['loan_id', 'reporting_period', 'current_upb', 'current_interest_rate', 'loan_age', 'remaining_months', 'delinquency_status']
    monthly_df = df[monthly_cols].copy()
    
    # Create zero balance code (dummy it for now if not clear)
    # usually col_42 is zero balance code, let's just make it up based on UPB
    monthly_df['zero_balance_code'] = np.where(monthly_df['current_upb'] == 0, '01', '')
    
    # Add target labels (dummy target: default if delinquency > 3, prepay if UPB=0)
    monthly_df['delinquency_status'] = pd.to_numeric(monthly_df['delinquency_status'], errors='coerce').fillna(0)
    monthly_df['target_default'] = (monthly_df['delinquency_status'] > 3).astype(int)
    monthly_df['target_prepay'] = (monthly_df['current_upb'] == 0).astype(int)
    
    # Split into train and test based on reporting period
    # Let's see unique periods
    periods = sorted(monthly_df['reporting_period'].dropna().unique())
    if len(periods) > 1:
        test_period = periods[-1] # use last month as test
        train_df = monthly_df[monthly_df['reporting_period'] != test_period]
        test_df = monthly_df[monthly_df['reporting_period'] == test_period].drop(columns=['target_default', 'target_prepay'])
    else:
        # random split
        train_df = monthly_df.sample(frac=0.8, random_state=42)
        test_df = monthly_df.drop(train_df.index).drop(columns=['target_default', 'target_prepay'])
        
    train_df.to_csv("loan_monthly_performance_train.csv", index=False)
    test_df.to_csv("loan_monthly_performance_test.csv", index=False)
    
    print("Extracting Servicer Updates...")
    # Generate some mock conflicting updates
    servicer_df = df[['loan_id', 'reporting_period', 'delinquency_status', 'current_upb']].sample(frac=0.05, random_state=42).copy()
    servicer_df['delinquency_status'] = servicer_df['delinquency_status'] + 1 # create conflict
    servicer_df['update_date'] = '2025-04-01'
    servicer_df.to_csv("servicer_updates.csv", index=False)
    
    print("Data processing complete.")

if __name__ == "__main__":
    process_data()
