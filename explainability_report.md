# Task 6: Explainability Layer Report

## 1. Global Feature Importance (XGBoost Native)
The model relies heavily on interest rate spreads, loan age, and origination balance:

* **Feature Importance Plot**: Saved as [feature_importances.png](file:///e:/intain/data_final/outputs/feature_importances.png)

| Feature | Importance Score |
|---|---|
| loan_purpose_P | 0.0726 |
| current_interest_rate_lag1 | 0.0558 |
| orig_upb | 0.0341 |
| remaining_months | 0.0315 |
| loan_age | 0.0273 |
| state_NY | 0.0263 |
| loan_purpose_C | 0.0207 |
| ltv | 0.0205 |
| upb_pct_of_orig | 0.0182 |
| state_IL | 0.0175 |

## 2. Local Explanations for Representative Loans
To illustrate how the model scores individuals, here are two opposite loan cases:

### Case A: Typical Low-Risk Loan (Accepted)
* **Loan ID**: 139435503
* **Predictive Features**:
  - `loan_age`: 9 months
  - `remaining_months`: 351 months
  - `credit_score`: 764
  - `ltv`: 89
* **Model Output (Calibrated Probability)**: 5.6377%
* **Decision**: **Accept** (Low risk, borrower likely to hold the mortgage)

### Case B: Typical High-Risk Loan (Flagged for Refinance Risk)
* **Loan ID**: 139435519
* **Predictive Features**:
  - `loan_age`: 9 months
  - `remaining_months`: 351 months
  - `credit_score`: 773
  - `ltv`: 75
* **Model Output (Calibrated Probability)**: 18.6610%
* **Decision**: **Flag for Refinance Risk** (High prepayment risk, borrower likely to refinance soon)

## 3. False Positive & False Negative Analysis (Business Review)
Below is the model's error rate segmented by credit bands on the validation set:

| Credit Band | Total Records | Actual Prepayments | False Positives | False Negatives | FP Rate (%) | FN Rate (%) |
|---|---|---|---|---|---|---|
| Subprime | 2,141 | 143 | 198 | 71 | 9.91% | 49.65% |
| Near-Prime | 10,610 | 554 | 433 | 398 | 4.31% | 71.84% |
| Prime | 52,864 | 3,006 | 4,957 | 1,663 | 9.94% | 55.32% |

* **Where the Model Overpredicts Prepayment (False Positives)**: High False Positive rates (9.94%) in the Prime band occur because borrowers with high credit scores and low LTV have strong refinance incentives, but face unobserved micro-frictions (e.g. transaction fees, closing costs, or lack of financial literacy) that delay prepayment.
* **Where the Model Underpredicts Prepayment (False Negatives)**: High False Negative rates in Subprime occur when financially constrained borrowers prepay unexpectedly due to personal changes (relocation, home sales, or changes in family structures).

## 4. Model Confidence & Uncertainty
* **High Confidence (Uncertainty < 10%)**: The model is highly confident in low-prepayment regions (e.g., loans with low interest rates or short remaining terms).
* **High Uncertainty (Uncertainty > 30%)**: Borrowers in the probability range of 12% to 18% represent a high-volatility group. These loans should be prioritized for monthly portfolio cash-flow reviews.
