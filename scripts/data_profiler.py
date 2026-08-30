import pandas as pd
import numpy as np
import json
import os
import io

def generate_profiling_report():
    print("Loading data...")
    try:
        train_df = pd.read_csv("loan_monthly_performance_train.csv")
        test_df = pd.read_csv("loan_monthly_performance_test.csv")
        static_df = pd.read_csv("loan_static_attributes.csv")
        servicer_df = pd.read_csv("servicer_updates.csv")
        with open("validation_rules.json") as f:
            validation_rules = json.load(f)
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    print("Profiling data...")
    
    # 1. Missingness
    missing_train = train_df.isnull().sum()
    missing_train = missing_train[missing_train > 0] / len(train_df)
    missing_static = static_df.isnull().sum()
    missing_static = missing_static[missing_static > 0] / len(static_df)
    
    # 2. Outliers (using IQR for numeric columns)
    num_cols = train_df.select_dtypes(include=[np.number]).columns
    outliers_summary = {}
    for col in num_cols:
        Q1 = train_df[col].quantile(0.25)
        Q3 = train_df[col].quantile(0.75)
        IQR = Q3 - Q1
        if IQR > 0:
            outliers = train_df[(train_df[col] < (Q1 - 1.5 * IQR)) | (train_df[col] > (Q3 + 1.5 * IQR))]
            if len(outliers) > 0:
                outliers_summary[col] = len(outliers) / len(train_df)

    # 3. Train-Test Drift (using basic mean/std differences for simplicity)
    drift_summary = {}
    for col in num_cols:
        if col in test_df.columns:
            train_mean = train_df[col].mean()
            test_mean = test_df[col].mean()
            if train_mean != 0 and pd.notnull(train_mean) and pd.notnull(test_mean):
                pct_diff = abs(train_mean - test_mean) / abs(train_mean)
                if pct_diff > 0.1: # >10% drift
                    drift_summary[col] = pct_diff
                    
    # 4. Data Quality Checks (from validation_rules)
    dq_scores = {}
    # Rule: current_upb <= orig_balance * 1.05
    # To check this, we need to join train_df and static_df
    merged_df = train_df.merge(static_df[['loan_id', 'orig_upb']], on='loan_id', how='left')
    balance_violations = merged_df[merged_df['current_upb'] > merged_df['orig_upb'] * 1.05]
    dq_scores['balance_consistency_violations'] = len(balance_violations)
    
    # Rule: delinquency_status cannot jump more than 1
    # Need to group by loan_id and sort by reporting_period
    sorted_df = train_df.sort_values(['loan_id', 'reporting_period'])
    sorted_df['prev_delinquency'] = sorted_df.groupby('loan_id')['delinquency_status'].shift(1)
    jump_violations = sorted_df[sorted_df['delinquency_status'] > sorted_df['prev_delinquency'] + 1]
    dq_scores['delinquency_progression_violations'] = len(jump_violations)
    
    # 5. Leakage Risk
    # Any column highly correlated with target_default or target_prepay
    leakage = []
    target_cols = ['target_default', 'target_prepay']
    for t_col in target_cols:
        if t_col in train_df.columns:
            for col in num_cols:
                if col != t_col:
                    corr = train_df[t_col].corr(train_df[col])
                    if abs(corr) > 0.6:
                        leakage.append(f"{col} correlates {corr:.2f} with {t_col}")

    # 6. Servicer Updates Conflict Detection
    servicer_merged = train_df.merge(servicer_df, on=['loan_id', 'reporting_period'], suffixes=('', '_servicer'))
    conflicts = servicer_merged[servicer_merged['delinquency_status'] != servicer_merged['delinquency_status_servicer']]
    
    # Generate the Markdown Report
    # We will write this out to standard output, which will be captured
    
    report = f"""# Task 1: Data Intelligence and Profiling Report

## 1. File-by-File Understanding
* **`loan_static_attributes.csv`**: Contains {len(static_df)} origination records. Grain is one row per loan.
* **`loan_monthly_performance_train.csv`**: Contains {len(train_df)} panel records. Grain is one row per loan per month. Includes targets (`target_default`, `target_prepay`).
* **`loan_monthly_performance_test.csv`**: Contains {len(test_df)} panel records (unlabeled).
* **`servicer_updates.csv`**: Contains {len(servicer_df)} updates representing secondary source data for conflict resolution.
* **`validation_rules.json`**: Defines deterministic checks for data integrity.
* **`data_dictionary.md`**: Provides plain-English field definitions.

## 2. Missingness Summary
**Static Attributes (>0% missing):**
{chr(10).join([f"* {k}: {v:.2%}" for k, v in missing_static.items()]) if not missing_static.empty else "No missing values found."}

**Monthly Performance (>0% missing):**
{chr(10).join([f"* {k}: {v:.2%}" for k, v in missing_train.items()]) if not missing_train.empty else "No missing values found."}

## 3. Outlier and Anomaly Summary
*Percentage of records outside 1.5 IQR bounds:*
{chr(10).join([f"* {k}: {v:.2%}" for k, v in outliers_summary.items()]) if outliers_summary else "No significant outliers found."}

## 4. Train-Test Drift Summary
*Features with >10% drift in mean values:*
{chr(10).join([f"* {k}: {v:.1%} difference" for k, v in drift_summary.items()]) if drift_summary else "No significant drift detected."}

## 5. Relationship Breaks and Validation Rules
* **Balance Consistency**: Found {dq_scores['balance_consistency_violations']} violations where Current UPB > Original Balance * 1.05.
* **Delinquency Progression**: Found {dq_scores['delinquency_progression_violations']} violations where delinquency status jumped more than 1 month at a time.
* **Servicer Conflicts**: Found {len(conflicts)} conflicting updates out of {len(servicer_merged)} joined records.

## 6. Leakage Risk Summary
The following fields are highly correlated with the target labels and pose a risk of data leakage (especially if they encode future state or are derived from the target):
{chr(10).join([f"* {l}" for l in leakage]) if leakage else "No high correlations (>0.6) with target labels found."}

## 7. Data-Quality Scoring Approach
* **Record-Level**: Assign a binary flag (0/1) for each validation rule violated (e.g., balance check, delinquency jump, missing critical fields). The score is `1 - (violations / total_rules)`.
* **Batch-Level**: Aggregate record-level scores. File score = `total passing records / total records`. A score below 95% indicates a critical batch failure.
* **Current Batch Score (Estimated)**: Based on {len(jump_violations) + len(balance_violations)} violations in {len(train_df)} rows, the estimated score is {1 - (len(jump_violations) + len(balance_violations))/len(train_df):.2%}.

## 8. Top Issues to Fix Before Modeling (Ranked by Severity)
1. **Servicer Conflicts**: High number of mismatched delinquency statuses requires a resolution logic (e.g., trust primary source vs secondary).
2. **Leakage Variables**: Remove fields like `delinquency_status` (if predicting default) that trivially give away the answer if they represent current state rather than a lagged state.
3. **Data Integrity**: Filter or cap the {dq_scores['balance_consistency_violations']} rows with impossible balance dynamics.
4. **Missing Values**: Impute missing variables (e.g., LTV, DTI) using median or model-based imputation.

## 9. Handoff Note for Task 2
**To the Modeling Team:** 
The datasets have been generated and profiled. You must ensure you lag the `delinquency_status` by 1 month to avoid leakage when predicting default. Address the servicer conflicts by creating a `resolved_delinquency` feature. Impute the missing values in static attributes before feeding them into the ML models. Good luck!
"""
    
    with open("report_output.txt", "w") as f:
        f.write(report)
    print("Profiling complete. Output saved to report_output.txt")

if __name__ == "__main__":
    generate_profiling_report()
