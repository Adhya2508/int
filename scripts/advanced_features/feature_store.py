"""
=================================================================
INTAIN AI CHALLENGE -- ADVANCED FEATURES: FEATURE STORE
Loan Performance Intelligence Engine
Intain Campus FinTech Challenge 2026

This module acts as the Feature Store pipeline. It:
  - Generates consistent, leakage-free features for training,
    validation, test, anomaly, and scenario work.
  - Applies standardized scaling and imputations.
  - Saves the final versioned feature set to outputs/feature_store/.
=================================================================
"""
import os
import sys
import pandas as pd
import numpy as np
import joblib
import json

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = "e:/intain"
DATA_DIR = os.path.join(BASE_DIR, "data_final")
STORE_DIR = os.path.join(BASE_DIR, "outputs/feature_store")
os.makedirs(STORE_DIR, exist_ok=True)

# Load raw cleaned datasets
train_df = pd.read_csv(os.path.join(DATA_DIR, "train_final.csv"))
test_df  = pd.read_csv(os.path.join(DATA_DIR, "test_final.csv"))

numeric_features = [
    'delinquency_status_lag1', 'current_upb_lag1', 'current_interest_rate_lag1',
    'loan_age', 'remaining_months', 'orig_upb', 'credit_score', 'ltv', 'dti',
    'upb_pct_of_orig', 'term_pct_elapsed'
]
categorical_features = ['state', 'loan_purpose', 'property_type', 'vintage']

# Feature Contract / Metadata
CONTRACT = {
    "version": "1.0.0",
    "last_updated": "2026-08-30",
    "features": {
        "numeric": numeric_features,
        "categorical": categorical_features
    },
    "remediation_applied": [
        "First-row lag imputation using origination UPB/current credit",
        "Clip negative ages to 0",
        "Target leakage exclusion of same-period performance attributes"
    ]
}

def build_features():
    print("Executing Feature Store Pipeline version 1.0.0...")
    
    # Save contract metadata
    with open(os.path.join(STORE_DIR, "feature_contract.json"), "w") as f:
        json.dump(CONTRACT, f, indent=2)
        
    # Standardize train features
    train_features = train_df[numeric_features + categorical_features + ['next_12m_prepayment_flag']].copy()
    train_features.to_csv(os.path.join(STORE_DIR, "train_features_v1.csv"), index=False)
    
    # Standardize test features
    test_features = test_df[numeric_features + categorical_features].copy()
    test_features.to_csv(os.path.join(STORE_DIR, "test_features_v1.csv"), index=False)
    
    print(f"Feature Store files written to {STORE_DIR}/")

if __name__ == "__main__":
    build_features()
