# 🎬 5-Minute Video Demo Flow Script
**Loan Performance Intelligence Engine**
*Intain Campus FinTech Challenge 2026 | AI Track*

---

## Overview & Demo Checklist
This 5-minute video demo script is structured around the **15 mandatory presentation checkpoints** for final quality-control review.

```
[0:00 - 0:40] Part 1: Dataset, Profiling, Quality & Feature Split (Checkpoints 1–5)
[0:40 - 1:40] Part 2: Model Benchmarks, Survival & Anomaly Detection (Checkpoints 6–9)
[1:40 - 2:40] Part 3: Stress Scenarios, Local Explainability & Counterfactuals (Checkpoints 10–11)
[2:40 - 3:50] Part 4: Grounded LLM Copilot, Rejection Audit & Submissions (Checkpoints 12–14)
[3:50 - 5:00] Part 5: AI Development Evidence Log & Closing Summary (Checkpoint 15)
```

---

## ⏱️ Minute-by-Minute Script & Visual Guide

### PART 1: Data Infrastructure & Engineering (Checkpoints 1–5)
**Duration:** 0:00 – 0:40 (40 seconds)

* **1. Dataset and Targets:**
  * *Speaker Narration:* "Welcome to the Loan Performance Intelligence Engine. Our input data consists of static origination attributes, monthly servicing panels, and servicer updates. The target is binary voluntary prepayment within the reporting horizon, extended into a 3-state competing risk hazard."
  * *Visual:* Open `data_dictionary.md` in VS Code, highlighting `loan_static_attributes.csv` and `loan_monthly_performance_*.csv`.

* **2. Data Profiling Report:**
  * *Speaker Narration:* "Task 1 data profiling established a 100% complete baseline across static attributes and audited feature cardinality across 10 longitudinal months."
  * *Visual:* Open `reports/data_intelligence_report.md` showing distribution summaries.

* **3. Top Data-Quality Issues:**
  * *Speaker Narration:* "We identified 3 critical DQ issues: missing servicer update timestamps, zero-balance code mismatches, and loan age jumps exceeding 1 month per reporting cycle."
  * *Visual:* Scroll to the Data Quality Audit table in `reports/data_intelligence_report.md`.

* **4. Feature-Engineering Approach:**
  * *Speaker Narration:* "Our feature store constructs 24 dynamic predictors including interest rate spreads, lagged UPB changes, delinquency progression vectors, and LTV-DTI interaction indices."
  * *Visual:* Open `scripts/advanced_features/feature_store.py` showing feature transforms.

* **5. Time-Aware Split:**
  * *Speaker Narration:* "Crucially, random train-test splits leak identical loans across longitudinal months. We enforced a strict chronological split: training on longitudinal history (months 0–10) and validating on a December 2025 cross-sectional snapshot (month 11)."
  * *Visual:* Highlight the split logic in `scripts/run_all_tasks.py`.

---

### PART 2: ML Models, Survival & Anomaly Detection (Checkpoints 6–9)
**Duration:** 0:40 – 1:40 (60 seconds)

* **6. Baseline Model Performance:**
  * *Speaker Narration:* "Our empirical baseline hazard model achieved an ROC-AUC of 0.5000 and a baseline Brier calibration score of 0.0141."
  * *Visual:* Open `reports/survival_report.md` baseline metrics table.

* **7. Improved Model Performance:**
  * *Speaker Narration:* "Our optimized XGBoost classifier achieved a validation ROC-AUC of **0.8116**, with a 95% bootstrap confidence interval of **[0.8037, 0.8195]**. Applying Platt scaling reduced raw calibration Brier loss from 0.3059 down to **0.0139**."
  * *Visual:* Point to the ROC-AUC and Brier Score cards in the interactive dashboard `dashboard.html`.

* **8. Survival or Transition Model Output:**
  * *Speaker Narration:* "We implemented a discrete-time competing risk survival model that estimates Cumulative Incidence Functions (CIF) for voluntary Prepayment versus Maturity, treating active month 12 loans as right-censored."
  * *Visual:* Click the "Competing Risk Model" tab in the interactive dashboard showing the CIF curves.

* **9. Anomaly Examples:**
  * *Speaker Narration:* "Using unsupervised Isolation Forests, we flagged 20 diverse exception loans, including Loan ID 139436802 which exhibited an anomaly score of 0.9500 due to severe historical term inconsistency."
  * *Visual:* Open `reports/anomaly_report.md` showing the flagged anomaly list.

---

### PART 3: Stress Scenarios & Local Explainability (Checkpoints 10–11)
**Duration:** 1:40 – 2:40 (60 seconds)

* **10. Scenario Output:**
  * *Speaker Narration:* "Under a -50 FICO credit shock, portfolio prepayment probability shifts downward, while a +2.0% interest rate hike reduces prepayment probability by 1.8%. Our 50-path Monte Carlo portfolio simulation projects a median 12-month prepayment rate of 5.97%."
  * *Visual:* Click the "Summary Metrics" tab in the dashboard showing the Monte Carlo and Stress Sensitivity charts.

* **11. Local Explanation for One Loan:**
  * *Speaker Narration:* "Task 6 provides SHAP local attributions. For Loan 139435505, a high LTV ratio of 89% and interest rate spread were the primary drivers pushing prepayment probability to 22.88%. We also generate counterfactual risk-reversal templates."
  * *Visual:* Open `reports/explainability_report.md` showing the SHAP force plot breakdown.

---

### PART 4: Grounded LLM Copilot & Rejection Auditing (Checkpoints 12–14)
**Duration:** 2:40 – 3:50 (70 seconds)

* **12. LLM-Generated Reviewer Note:**
  * *Speaker Narration:* "Task 7 builds an LLM Reviewer Copilot powered by OpenAI `gpt-4o-mini`. The LLM does NOT make final decisions or replace ML models; it reads structured model outputs and drafts grounded reviewer recommendations with mandatory disclaimers."
  * *Visual:* Open `outputs/copilot_examples/copilot_note_139435503.json` in VS Code showing the generated note.

* **13. Example of LLM Output Rejected or Corrected:**
  * *Speaker Narration:* "To enforce strict governance, our guardrails catch ungrounded LLM drafts. Here in `rejected_examples.jsonl`, an LLM output attempting to make a final application rejection ('decided to reject application') was intercepted and corrected to a grounded recommendation ('review requested')."
  * *Visual:* Open `logs/copilot/rejected_examples.jsonl` highlighting `reason_for_rejection`.

* **14. Final Submission File:**
  * *Speaker Narration:* "The full pipeline writes the final challenge submission file containing calibrated prepayment probabilities for all test records."
  * *Visual:* Open `outputs/predictions/submission.csv` (first 10 rows).

---

### PART 5: AI Development Evidence Log & Summary (Checkpoint 15)
**Duration:** 3:50 – 5:00 (70 seconds)

* **15. AI Development Log:**
  * *Speaker Narration:* "Task 8 records our full Agentic AI Development Log. Every LLM prompt is hashed in `logs/copilot/prompts/`, model hyperparameters are logged in `experiment_registry.jsonl`, and RAG search logs are stored in `rag_log.jsonl` for 100% auditability."
  * *Visual:* Open `logs/experiment_tracking/experiment_registry.jsonl` and show the prompt SHA256 hashes in `logs/copilot/prompts/`.

* **Closing Summary:**
  * *Speaker Narration:* "In summary, the Loan Performance Intelligence Engine combines rigorous ML modeling, time-aware validation, survival hazard estimation, grounded LLM copilot recommendations, and complete auditability. Thank you."
  * *Visual:* Return to the main dashboard `dashboard.html`.

---

## 📋 Pre-Recording Verification Checklist

| # | Checkpoint | Location in Repository / UI | Status |
|---|------------|-----------------------------|--------|
| 1 | Dataset & targets | `data_dictionary.md` | ✅ Verified |
| 2 | Data profiling report | `reports/data_intelligence_report.md` | ✅ Verified |
| 3 | Top data-quality issues | `reports/data_quality_audit_report.md` | ✅ Verified |
| 4 | Feature-engineering approach | `scripts/advanced_features/feature_store.py` | ✅ Verified |
| 5 | Time-aware split | `scripts/run_all_tasks.py` (months 0-10 vs 11) | ✅ Verified |
| 6 | Baseline model performance | `reports/survival_report.md` (AUC=0.5000) | ✅ Verified |
| 7 | Improved model performance | `dashboard.html` (AUC=0.8116, Brier=0.0139) | ✅ Verified |
| 8 | Survival / transition model | `dashboard.html` (Competing Risk tab) | ✅ Verified |
| 9 | Anomaly examples | `reports/anomaly_report.md` (Top 20 loans) | ✅ Verified |
| 10 | Scenario output | `reports/scenario_report.md` & Monte Carlo | ✅ Verified |
| 11 | Local SHAP explanation | `reports/explainability_report.md` | ✅ Verified |
| 12 | LLM reviewer note | `outputs/copilot_examples/copilot_note_*.json` | ✅ Verified |
| 13 | Rejected LLM output example | `logs/copilot/rejected_examples.jsonl` | ✅ Verified |
| 14 | Final submission file | `outputs/predictions/submission.csv` | ✅ Verified |
| 15 | AI Development Log | `logs/experiment_tracking/` & prompt hashes | ✅ Verified |
