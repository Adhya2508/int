# Task 1: Data Intelligence & Profiling Report

## 1. Shape, Schema, and Types
* **Train Set (`train_final.csv`)**: 374,275 rows, 20 columns
  - Panel grain: loan-month.
  - Features: 11 numeric, 4 categorical.
* **Test Set (`test_final.csv`)**: 32,176 rows, 18 columns
* **Static Attributes (`static_final.csv`)**: 34,758 rows, 9 columns

## 2. Missingness Summary
* **Train Missingness**:
  - loan_age: 0.69%
  - remaining_months: 0.69%
  - credit_score: 0.17%
  - term_pct_elapsed: 0.69%
* **Test Missingness**:
  - current_interest_rate_lag1: 0.03%
  - loan_age: 1.57%
  - remaining_months: 1.57%
  - credit_score: 0.18%
  - upb_pct_of_orig: 0.03%
  - term_pct_elapsed: 1.57%

## 3. Duplicate Checks
* **Train Set Duplicate (loan_id, reporting_period) keys**: 0 (Target: 0)
* **Test Set Duplicate (loan_id, reporting_period) keys**: 0 (Target: 0)
* **Static Attributes Duplicate loan_id keys**: 0 (Target: 0)

## 4. Outlier Summary
*Percentage of rows outside 1.5 IQR bounds (Train set):*
  - current_upb_lag1: 1.82%
  - current_interest_rate_lag1: 1.43%
  - loan_age: 0.00%
  - remaining_months: 8.41%
  - orig_upb: 1.03%
  - credit_score: 1.31%
  - ltv: 1.26%
  - dti: 0.75%
  - upb_pct_of_orig: 7.58%
  - term_pct_elapsed: 1.21%

## 5. Invalid Date Relationships
* Train set max reporting date: 2025-11-01
* Test set min reporting date: 2025-12-01
* **Chronological overlap check**: PASS (strictly sequential)

## 6. Correlation / Highly Dependent Fields
* High correlation (>0.6) with `next_12m_prepayment_flag`:
  - `upb_pct_of_orig`: 0.79
  - `term_pct_elapsed`: 0.65
  - All retired targets and same-row performance variables have been excluded, eliminating tautological leakage.

## 7. Cross-Column Rule Violations (from validation_rules.json)
* **Balance Consistency** (`current_upb_lag1 <= orig_upb * 1.05`): 0 violations.
* **Delinquency Progression** (`dlq(t) <= dlq(t-1) + 1`): 0 violations.

## 8. Record and Batch Quality Scores
* **Sample Record-level Quality Score (Mean)**: 100.0000%
* **Sample Record-level Quality Score (Median)**: 100.0000%
* **Batch-level Quality Score**: 100.0000% (Above 95% target, PASS)
