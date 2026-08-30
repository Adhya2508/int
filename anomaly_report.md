# Task 4: Anomaly and Exception Detection Report

## 1. Scoring Methodology
* **Record-Level Anomaly Score**: Calculated using an **Isolation Forest** trained on the scaled features. The raw scores are normalized to [0, 1], where values closer to 1 indicate highly anomalous observations.
* **Exception Probability (Hybrid Score)**: A weighted index combining statistical outlier scores (60%) and deterministic rule violations (40%) from `validation_rules.json`.
* **Unique Filter**: To make this report highly actionable for human reviewers, we group exceptions by `loan_id` and pick the most anomalous month. This ensures 20 distinct loan accounts are shown, rather than repeating the same loan multiple times.

## 2. Summary of Flagged Exception Categories
The top exceptions are classified into the following types:

| Anomaly Type | Count in Top 20 |
|---|---|
| Statistical Outlier | 10 |
| Severe Term Inconsistency (Quarantined) | 5 |
| Subprime Credit Attribute | 3 |
| High Debt-to-Income | 2 |

## 3. Reviewer-Ready Anomaly Examples (Top 20 Unique Suspicious Loans)
The following records are flagged as exceptions and should be manually reviewed:

| Loan ID | Period | Primary Exception Category | Exception Score / Flag | Reviewer Investigation Note |
|---|---|---|---|---|
| 139458706 | 112025 | Statistical Outlier | 0.6000 | Flags extreme multivariate combination of UPB, remaining term, and loan age. |
| 139437973 | 112025 | Statistical Outlier | 0.5322 | Flags extreme multivariate combination of UPB, remaining term, and loan age. |
| 139436423 | 112025 | Statistical Outlier | 0.5139 | Flags extreme multivariate combination of UPB, remaining term, and loan age. |
| 139461096 | 112025 | Statistical Outlier | 0.5125 | Flags extreme multivariate combination of UPB, remaining term, and loan age. |
| 139436771 | 102025 | Statistical Outlier | 0.5120 | Flags extreme multivariate combination of UPB, remaining term, and loan age. |
| 139456835 | 102025 | Statistical Outlier | 0.5005 | Flags extreme multivariate combination of UPB, remaining term, and loan age. |
| 139449694 | 102025 | Statistical Outlier | 0.5002 | Flags extreme multivariate combination of UPB, remaining term, and loan age. |
| 139435721 | 102025 | Statistical Outlier | 0.4972 | Flags extreme multivariate combination of UPB, remaining term, and loan age. |
| 139463965 | 112025 | Statistical Outlier | 0.4949 | Flags extreme multivariate combination of UPB, remaining term, and loan age. |
| 139465017 | 112025 | Statistical Outlier | 0.4913 | Flags extreme multivariate combination of UPB, remaining term, and loan age. |
| 139463026 | 62025 | Subprime Credit Attribute | 0.8850 | Origination credit score is 540, which is exceptionally low for this prime cohort. |
| 139443746 | 92025 | Subprime Credit Attribute | 0.8850 | Origination credit score is 562, which is exceptionally low for this prime cohort. |
| 139443490 | 92025 | Subprime Credit Attribute | 0.8850 | Origination credit score is 563, which is exceptionally low for this prime cohort. |
| 139440199 | 62025 | High Debt-to-Income | 0.8710 | Debt-to-income ratio is 62.0%, representing extreme borrower debt leverage. |
| 139459702 | 102025 | High Debt-to-Income | 0.8710 | Debt-to-income ratio is 54.0%, representing extreme borrower debt leverage. |
| 139436802 | ALL | Severe Term Inconsistency (Quarantined) | 1.0000 | Implied loan term varies severely over time. Isolated and quarantined during remediation. |
| 139445232 | ALL | Severe Term Inconsistency (Quarantined) | 1.0000 | Implied loan term varies severely over time. Isolated and quarantined during remediation. |
| 139445726 | ALL | Severe Term Inconsistency (Quarantined) | 1.0000 | Implied loan term varies severely over time. Isolated and quarantined during remediation. |
| 139452381 | ALL | Severe Term Inconsistency (Quarantined) | 1.0000 | Implied loan term varies severely over time. Isolated and quarantined during remediation. |
| 139453489 | ALL | Severe Term Inconsistency (Quarantined) | 1.0000 | Implied loan term varies severely over time. Isolated and quarantined during remediation. |

## 4. Detailed Explanations of Anomaly Drivers
1. **Balance Inconsistency**: Loans where current UPB exceeds origination balance by >5%. This represents a critical data entry error or unrecorded recapitalization event.
2. **Temporal Term Exceptions**: Negative loan age or remaining term represents a processing system error in date parsing.
3. **Statistical Outliers**: Isolation Forest isolates loans with extreme feature patterns (such as extremely low credit scores or DTI ratios exceeding normal thresholds).
