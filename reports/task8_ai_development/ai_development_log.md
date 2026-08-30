# Task 8: AI Development Log & Agentic ML Development Evidence

## 1. Objective
This document provides a detailed, judge-facing registry of how Artificial Intelligence (Antigravity AI Coding Agent and Groq LLM Endpoint) and human engineers collaborated to build, audit, and refine the **Loan Performance Intelligence Engine** for the Intain Campus FinTech Challenge 2026.

## 2. AI Tools Used
* **Antigravity AI Coding Agent:** Used as the primary pair-programming agent for directory auditing, python scripting, model training, plot generation, and automated report drafting.
* **Groq LLM API (`groq/compound-mini` / `llama-3.3-70b-versatile`):** Integrated via OpenAI SDK compatibility to generate natural language reviewer notes under Task 7.

## 3. Project Development Log (Tasks 1 to 7)

| Date / Time | Project Stage | AI Model / Tool | Prompt / Task | Output Summary | Accepted? | Human Edits & Corrections |
|---|---|---|---|---|---|---|
| 2026-08-30 | Task 1: Audit | Antigravity | Audit loan monthly, static, and servicer CSV files for leakage and quality. | Drafted `data_audit.py` checking missingness, dates, and rule hits. | **Partial** | Human verified and corrected the temporal chronological sorting code which sorted reporting periods as strings. |
| 2026-08-30 | Task 1: Clean | Antigravity | Write remediation pipeline to clean negative ages and impute lags. | Created `data_remediation.py` clipping negative age and imputing first-row lag nulls. | **Yes** | Accepted as-is; lags are filled with origination values (`orig_upb` and `0.0` delinquency). |
| 2026-08-30 | Task 2: Modeling | Antigravity | Train XGBoost prepayment classifier on validation splits. | Trained XGBoost with `scale_pos_weight` and Platt calibration. | **Yes** | Approved the chronological split boundary (train: Jan-Sep 2025; val: Oct-Nov 2025). |
| 2026-08-30 | Task 3: Survival | Antigravity | Build discrete-time survival model for prepayments. | Trained XGBoost hazard model but reported uncalibrated Brier Score of 0.3059 vs baseline 0.0141. | **Partial** | Human rejected the uncalibrated metrics. AI corrected the code by applying Platt scaling to the hazard model. |
| 2026-08-30 | Task 4: Anomaly | Antigravity | Flag suspicious records using Isolation Forest. | Ran Isolation Forest but output table was dominated by the same 3 loan IDs over time. | **Partial** | Human requested unique examples. AI grouped exceptions by `loan_id` and pulled 5 quarantined term drift records. |
| 2026-08-30 | Task 5: Scenario | Antigravity | Simulate 12-month projections under adverse/prepay stresses. | Simulated cohort projections and cumulative rates. | **Yes** | Human added a strict warning that delinquency/default stress is non-viable due to zero default positives in the vintage. |
| 2026-08-30 | Task 6: Explain | Antigravity | Output feature importances and error segment tables. | Output global importances and 1 local example. | **Partial** | Human added horizontal feature importance plot and a second opposite case study (high-risk vs low-risk). |
| 2026-08-30 | Task 7: Copilot | Antigravity | Implement LLM-Assisted Reviewer Copilot. | Built `copilot.py` calling Groq API to generate reviewer JSON. | **Yes** | Human enforced a custom dotenv loader to read secrets securely from `.env` in the project root. |

---

## 4. Accepted vs. Rejected Outputs

### Example 1: Accepted AI Code (Lag Imputation)
* **AI Output:** Suggestion to impute the first month of lag features (which are null due to shift operations) using origination values.
* **Code Block:**
  ```python
  df.loc[df['loan_age'] == 0, 'delinquency_status_lag1'] = 0.0
  df.loc[df['loan_age'] == 0, 'current_upb_lag1'] = df['orig_upb']
  ```
* **Human Action:** Accepted. This is mathematically correct and prevents the dropping of 34,747 early-stage observations.

### Example 2: Rejected AI Code (Same-Period Feature Leakage)
* **AI Output:** Suggestion to include `delinquency_status` and `current_upb` from the same month as features for prepayment modeling.
* **Human Action:** Rejected. Predicting prepayment using same-period outstanding balance is a post-event indicator (if the loan prepays, the UPB drops to 0, causing the model to learn a trivial association). The human corrected the features list to strictly use lagged features (`current_upb_lag1`, `delinquency_status_lag1`).

### Example 3: Corrected AI Report (Survival Brier Inconsistency)
* **AI Output (Rejected version):** Reported baseline Brier score of 0.0141 and XGBoost Brier score of 0.2948, but claimed the XGBoost model was superior.
* **Reason for Rejection:** A Brier score of 0.2948 is significantly worse (higher) than a baseline rate model of 0.0141. The claim of model superiority was false because the raw probabilities from `scale_pos_weight` were uncalibrated.
* **Corrected Version (Accepted):**
  ```python
  # Apply Platt Scaling
  calibrator_surv = LogisticRegression()
  calibrator_surv.fit(raw_val_surv_probs.reshape(-1, 1), y_val_surv)
  ```
  This calibrated the probabilities and dropped the XGBoost Brier score to **$0.013969$**, beating the baseline.

---

## 5. Approximate AI-Generated Code Share
* **AI-Generated / AI-Assisted Share:** ~85%
  - Drafted core data cleaning loops, sklearn preprocessors, XGBoost models, and Platt calibrators.
  - Drafted custom markdown report strings and markdown file exporters.
* **Human-Driven / Heavily Edited Share:** ~15%
  - Defined the chronological split parameters to prevent time-leakage.
  - Corrected f-string escaping and bracket parsing syntax errors.
  - Excluded the servicer dataset due to null delinquency and same-period UPB leakage.
  - Designed the hybrid statistical/business rule anomaly index and uniqueness grouping.

---

## 6. Human Review Process & Governance
1. **Model Guardrails:** Enforced that the LLM copilot is only an assistant drafting reviewer notes. The predictive scores and anomaly outputs are driven by the deterministic non-LLM XGBoost and Isolation Forest models.
2. **Leakage Audits:** Inspected all feature sets to guarantee no same-row performance variables were included in modeling.
3. **Traceability:** Double-logged prompts to `logs/prompts/` and executions to JSONL formats, enabling full offline audits.
4. **Safety Disclaimers:** Every LLM-generated note is parsed and checked to ensure it includes the analyst ownership disclaimer.

---

## 7. Lessons Learned
* **AI Strengths:** Incredibly fast script generation, data cleaning loop drafting, and report formatting.
* **AI Weaknesses:** High risk of mathematical errors (such as metric comparisons and f-string brace syntax) when generating latex formulas or handling highly imbalanced classifiers.
* **Takeaway:** AI pair-programming speeds up delivery by 10x, but rigorous human code review, validation split design, and metric verification are mandatory to deliver a robust financial engine.
