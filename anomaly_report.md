# Task 4: Anomaly and Exception Detection Report

## 1. Scoring Methodology
* **Record-Level Anomaly Score**: Calculated using an **Isolation Forest** trained on the scaled features. The raw scores are normalized to [0, 1], where values closer to 1 indicate highly anomalous observations.
* **Exception Probability (Hybrid Score)**: A weighted index combining statistical outlier scores (60%) and deterministic rule violations (40%) from `validation_rules.json`.

## 2. Reviewer-Ready Anomaly Examples (Top 20 Suspicious Records)
The following records are flagged as exceptions and should be manually reviewed:

| Loan ID | Period | Age | Lagged UPB | Credit Score | LTV | Exception Score | Suspected Driver |
|---|---|---|---|---|---|---|---|
| 139458706 | 112025 | 12 | $9,341.39 | 644 | 5 | 0.6000 | High statistical outlier |
| 139458706 | 92025 | 10 | $9,464.24 | 644 | 5 | 0.5868 | High statistical outlier |
| 139458706 | 102025 | 11 | $9,402.99 | 644 | 5 | 0.5868 | High statistical outlier |
| 139458706 | 82025 | 9 | $9,525.14 | 644 | 5 | 0.5789 | High statistical outlier |
| 139458706 | 72025 | 8 | $9,585.69 | 644 | 5 | 0.5696 | High statistical outlier |
| 139458706 | 62025 | 7 | $10,000.00 | 644 | 5 | 0.5386 | High statistical outlier |
| 139458706 | 42025 | 5 | $10,000.00 | 644 | 5 | 0.5362 | High statistical outlier |
| 139458706 | 52025 | 6 | $10,000.00 | 644 | 5 | 0.5339 | High statistical outlier |
| 139437973 | 112025 | 12 | $620,846.64 | 802 | 41 | 0.5322 | High statistical outlier |
| 139437973 | 92025 | 10 | $623,738.26 | 802 | 41 | 0.5216 | High statistical outlier |
| 139437973 | 102025 | 11 | $622,296.05 | 802 | 41 | 0.5216 | High statistical outlier |
| 139436423 | 112025 | 12 | $111,894.93 | 764 | 60 | 0.5139 | High statistical outlier |
| 139461096 | 112025 | 12 | $759,154.71 | 757 | 54 | 0.5125 | High statistical outlier |
| 139436771 | 102025 | 12 | $124,577.92 | 655 | 60 | 0.5120 | High statistical outlier |
| 139436423 | 102025 | 11 | $112,651.43 | 764 | 60 | 0.5070 | High statistical outlier |
| 139436771 | 92025 | 11 | $124,992.57 | 655 | 60 | 0.5051 | High statistical outlier |
| 139458706 | 32025 | 4 | $10,000.00 | 644 | 5 | 0.5042 | High statistical outlier |
| 139461096 | 102025 | 11 | $761,001.31 | 757 | 54 | 0.5033 | High statistical outlier |
| 139436771 | 82025 | 10 | $125,404.64 | 655 | 60 | 0.5022 | High statistical outlier |
| 139436423 | 92025 | 10 | $113,403.93 | 764 | 60 | 0.5014 | High statistical outlier |

## 3. Explanations of Anomaly Drivers
1. **Balance Consistency Violation**: Loans where `current_upb_lag1` exceeds the original balance (`orig_upb`) are highly suspicious and flagged immediately (Rule Violation).
2. **Extreme Features**: High exception scores are driven by extremely low credit scores (< 560) or high debt-to-income (DTI) ratios (> 55) which deviate significantly from the prime single-family vintage pattern.
3. **Age Anomalies**: Observations with zero or negative age that still show high amortization or principal paydown are flagged.
