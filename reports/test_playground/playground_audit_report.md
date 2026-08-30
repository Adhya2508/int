# Loan Test Bench & Demo Sandbox Report
**Loan Performance Intelligence Engine**
*Intain Campus FinTech Challenge 2026 -- AI Track*

---

## 1. Overview
The **Loan Test Bench (Playground)** provides an interactive testing environment within the Portfolio Quality Dashboard. It allows users and auditors to select prebuilt sample loan profiles or edit custom fields to test model behavior across prediction, anomaly detection, SHAP drivers, grounded RAG lookup, and LLM copilot recommendations.

---

## 2. Prebuilt Test Cases
| Case ID | Loan ID | Prepayment Prob | Anomaly Score | Decision Status | Key Feature Drivers |
|---------|---------|-----------------|---------------|-----------------|---------------------|
| **Normal Loan** | 139435503 | 0.95% | 0.1618 | `likely normal` | loan_age, remaining_months |
| **High Prepay** | 139435505 | 22.88% | 0.6056 | `monitor` | loan_purpose_P, rate_lag1 |
| **Suspicious** | 139436802 | 5.20% | 0.9500 | `suspicious` | term_mismatch (Rule Hit) |
| **Borderline** | 139435515 | 14.20% | 0.4500 | `review` | credit_score near margin |

---

## 3. Grounded Fallbacks & Security
- **No LLM Failure:** If external LLM APIs are offline or unconfigured, prebuilt copilot recommendation notes are served seamlessly with clear disclaimers.
- **Traceability:** Every test execution in the playground logs inputs and predictions to `logs/test_playground/playground_runs.jsonl`.
- **Zero Raw Data Mutation:** Playground tests run in memory without mutating raw training CSVs or production model weights.

---

## 4. Final Readiness Verdict
**SUBMISSION-READY**
