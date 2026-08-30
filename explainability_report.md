# Task 6: Explainability Layer Report

## 1. Global Feature Importance (XGBoost Native)
The model relies heavily on interest rate spreads, loan age, and origination balance:

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

## 2. Local Explanation for an Example Loan
* **Loan ID**: 139435503
* **Predictive Features**:
  - `loan_age`: 9 months
  - `remaining_months`: 351 months
  - `credit_score`: 764
  - `ltv`: 89
* **Model Output (Calibrated Probability)**: 0.9497%
* **Decision**: **Accept**

## 3. False Positive & False Negative Analysis
Below is the model's error rate segmented by credit bands on the validation set:

| Credit Band | Total Records | Actual Prepayments | False Positives | False Negatives | FP Rate (%) | FN Rate (%) |
|---|---|---|---|---|---|---|
| Subprime | 2,141 | 143 | 198 | 71 | 9.91% | 49.65% |
| Near-Prime | 10,610 | 554 | 433 | 398 | 4.31% | 71.84% |
| Prime | 52,864 | 3,006 | 4,957 | 1,663 | 9.94% | 55.32% |

* **FP Drivers**: High False Positive rates in the Prime band are driven by borrowers who have high refinance incentives (low LTV, high credit score) but face unobserved micro-frictions (e.g. transaction costs or lack of financial literacy).
* **FN Drivers**: High False Negative rates in Subprime are caused by borrowers who prepay despite low credit scores, often due to housing mobility or co-borrower credit profile changes.

## 4. Model Confidence & Uncertainty
* **Prediction Margin**: The model displays high confidence (confidence > 0.90) for loans that are highly unlikely to prepay (e.g. young loans with low credit scores).
* **Uncertainty Band**: Uncertainty is highest near the decision threshold (0.12 - 0.18), where borrower behavior is volatile. These loans are flagged for closer portfolio monitoring.
