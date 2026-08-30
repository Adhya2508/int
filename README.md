# 🏦 Loan Performance Intelligence Engine
**Intain Campus FinTech Challenge 2026 — AI Track**

> A production-grade, end-to-end Loan Performance Intelligence System featuring predictive modeling, anomaly detection, survival analysis, scenario simulation, LLM-assisted review copilot, and an interactive analytics dashboard.

---

## 📌 Project Overview

This project implements a complete AI-powered loan performance analytics pipeline for structured mortgage portfolios. It processes loan-level data across monthly performance panels, applies statistical models for prepayment prediction and anomaly detection, and surfaces results through an interactive dashboard and a grounded LLM reviewer copilot.

**Challenge Track:** Intain Campus FinTech Challenge 2026 — AI Track  
**Team / Author:** Adhya  
**Model AUC:** 0.8116 | **Brier Score (Calibrated):** 0.0139 | **PSI Drift Alert:** loan_age (expected chronological shift)

---

## 🗂️ Project Structure

```
e:/intain/
│
├── data/                           # Raw input data (DO NOT MODIFY)
│   ├── loan_static_attributes.csv
│   ├── loan_monthly_performance_*.csv
│   └── servicer_updates.csv
│
├── scripts/                        # All Python pipeline scripts
│   ├── run_all_tasks.py            # Master pipeline: Tasks 1–6
│   ├── task7_copilot.py            # LLM reviewer copilot (Task 7)
│   ├── task8_ai_dev_log.py         # AI development evidence log (Task 8)
│   └── advanced_features/
│       ├── run_advanced_features.py    # Advanced metric calculations
│       ├── generate_dashboard.py       # HTML dashboard compiler
│       ├── rag_assistant.py            # Grounded RAG search engine
│       ├── agentic_runner.py           # Full pipeline agentic runner
│       └── feature_store.py           # Versioned feature store
│
├── outputs/                        # Model outputs and analysis results
│   ├── advanced_features/          # Competing risk, active learning, stress
│   ├── calibration/                # Per-segment Brier score calibration
│   ├── confidence_intervals/       # Bootstrap AUC confidence intervals
│   ├── copilot_examples/           # LLM reviewer copilot outputs
│   ├── counterfactuals/            # Counterfactual loan risk scenarios
│   ├── fairness/                   # Segment-level TPR fairness metrics
│   ├── monitoring/                 # PSI covariate drift metrics
│   ├── monte_carlo/                # Monte Carlo portfolio simulation
│   └── predictions/                # Final model predictions & submission
│
├── reports/                        # Human-readable Markdown reports
│   ├── data_intelligence_report.md
│   ├── survival_report.md
│   ├── anomaly_report.md
│   ├── scenario_report.md
│   ├── explainability_report.md
│   └── dashboard/
│       └── dashboard_audit_report.md
│
├── dashboard/
│   └── advanced_features/
│       └── dashboard.html          # 📊 Main interactive dashboard (open in browser)
│
├── logs/                           # Audit and traceability logs
│   ├── copilot/
│   ├── rag/
│   └── experiment_tracking/
│
├── models/                         # Saved trained model artifacts
│
├── data_dictionary.md              # Field definitions for all datasets
├── validation_rules.json           # Business validation rule constraints
├── .env.example                    # Environment variable template
└── README.md                       # This file
```

---

## 🚀 Quick Start

### 1. Prerequisites

```bash
pip install pandas scikit-learn xgboost lifelines shap openai python-dotenv
```

### 2. Set up environment

```bash
cp .env.example .env
# Edit .env and add your GROQ API key:
# OPENAI_API_KEY=gsk_your_key_here
```

### 3. Run the full pipeline

```bash
# Run Tasks 1–8 + advanced features + dashboard in one command
py scripts/advanced_features/agentic_runner.py
```

### 4. Open the dashboard

```bash
# Open dashboard/advanced_features/dashboard.html in your browser
start dashboard/advanced_features/dashboard.html
```

---

## 📊 Dashboard Tabs

| Tab | What it shows |
|-----|--------------|
| **Summary Metrics** | Monte Carlo prepayment rate, ROC-AUC, 95% bootstrap CI, synthetic stress rate |
| **Competing Risk Model** | Cumulative Incidence Functions for Prepayment vs Maturity, survival probability S(t) |
| **Drift Monitoring** | PSI covariate drift scores per feature, ranked severity table |
| **Fairness & Bias** | Segment-level TPR (Equal Opportunity) and Brier calibration by credit band |
| **Active Learning Queue** | Top-10 uncertain loans ranked by priority score for human review |
| **Grounded RAG Search** | Keyword search over data dictionary & validation rules, plain-English summaries |

---

## 🧠 Tasks Implemented

### Task 1 — Data Intelligence & Profiling
- Missing value analysis, data type audit, feature distribution profiles
- Output: `reports/data_intelligence_report.md`

### Task 2 — Prepayment Prediction Model
- XGBoost classifier trained on longitudinal loan panel
- Chronological train/validation split to prevent leakage
- **Validation ROC-AUC: 0.8116**
- Output: `outputs/predictions/submission.csv`

### Task 3 — Survival & Competing Risk Analysis
- Discrete-time hazard model approximating Kaplan-Meier survival curves
- Competing risk CIF for Prepayment vs Maturity events
- Platt scaling calibration: Brier score reduced from 0.3059 → **0.0139**
- Output: `reports/survival_report.md`, `outputs/advanced_features/competing_risk.json`

### Task 4 — Anomaly & Exception Detection
- Isolation Forest unsupervised anomaly detection
- 20 diverse flagged loan examples with anomaly type labels
- Output: `reports/anomaly_report.md`

### Task 5 — Scenario & Stress Simulation
- Base / Stress / Severe stress scenarios across credit bands, states, vintages
- Monte Carlo portfolio simulation over 12 months (50 paths)
- Output: `reports/scenario_report.md`, `outputs/monte_carlo/`

### Task 6 — Explainability Layer
- SHAP global feature importance and local loan-level explanations
- Counterfactual risk reversal templates
- Output: `reports/explainability_report.md`, `outputs/counterfactuals/`

### Task 7 — LLM Reviewer Copilot
- Groq LLM-backed reviewer assistant grounded in model outputs
- Reads anomaly score, prepayment probability, SHAP drivers, validation rules
- Returns a concise reviewer note — **never makes a final decision**
- Fallback mock note if API rate-limited
- Output: `outputs/copilot_examples/`, `logs/copilot/`

### Task 8 — AI Development Evidence Log
- Full agentic development log showing AI tools used throughout the project
- Prompt-response traces, model experiment registry, iteration history
- Output: `logs/experiment_tracking/`

---

## 🔬 Advanced Features

| Feature | Description |
|---------|-------------|
| **Competing Risk Survival** | 3-state discrete-time hazard model (Survive / Prepaid / Matured) |
| **Monte Carlo Simulation** | 12-month portfolio-level prepayment path simulation |
| **Covariate Drift (PSI)** | Population Stability Index per feature, train vs test shift |
| **Platt Calibration** | Per-segment probability calibration reducing ECE error |
| **Fairness / Bias Audit** | Equal Opportunity TPR across Prime / Near-Prime / Subprime |
| **Active Learning Queue** | Uncertainty × Anomaly priority scoring for human review |
| **Grounded RAG Search** | Keyword retrieval over dictionary + rules, zero hallucination |
| **Bootstrap Confidence Intervals** | 95% CI on validation AUC via 20 resamples |
| **Stress Sensitivity Analysis** | Credit stress (−50 FICO) and rate stress (+2%) shift analysis |
| **Counterfactual Templates** | Hypothetical risk reversal scenarios per loan |

---

## ⚠️ Design Decisions & Audit Notes

### Chronological Split (No Leakage)
Training uses all records from months 1–10. Validation uses month 11 only. No loan crosses both sets, so there is no target leakage.

### PSI Drift Explanation
`loan_age` shows a high PSI (~18) because training is a longitudinal panel (ages 0–10) while test is a single cross-sectional snapshot (all loans at age 11). This is expected chronological aging, not a data quality defect.

### Calibration
Raw survival model hazard probabilities produce a Brier score of 0.3059 (worse than the 0.0141 empirical baseline). After Platt scaling, the score improves to **0.0139**, beating the baseline.

### LLM Role
The LLM copilot (Task 7) is a reviewer **assistant only**. It never makes credit decisions. All outputs are labeled as recommendations, grounded in model outputs and documentation.

### Fairness Framing
The bias audit compares model performance across credit score bands (Prime / Near-Prime / Subprime). This is a segment performance audit, not a formal ECOA/FHA fairness analysis (protected class variables are not present in the dataset).

---

## 🔑 Environment Variables

Copy `.env.example` to `.env` and fill in:

```env
OPENAI_API_KEY=gsk_your_groq_api_key_here   # Groq API key for Task 7 copilot
```

> `.env` is gitignored. Never commit secrets.

---

## 📈 Key Metrics Summary

| Metric | Value |
|--------|-------|
| Validation ROC-AUC | 0.8116 |
| AUC 95% Bootstrap CI | [0.8037, 0.8195] |
| Brier Score (calibrated) | 0.0139 |
| Brier Score (raw) | 0.3059 |
| Empirical Brier Baseline | 0.0141 |
| Monte Carlo Median Prepay Rate | ~6% |
| Synthetic Stress Prepay Rate | ~17% |
| Loans Flagged (Anomaly) | 20 diverse examples |
| Active Learning Queue | Top 10 uncertain loans |

---

## 🧪 Running Individual Scripts

```bash
# Task 1–6 master pipeline
py scripts/run_all_tasks.py

# Task 7 LLM copilot test
py scripts/task7_copilot.py

# Advanced features only
py scripts/advanced_features/run_advanced_features.py

# Regenerate dashboard HTML
py scripts/advanced_features/generate_dashboard.py

# RAG assistant test
py scripts/advanced_features/rag_assistant.py

# Full agentic pipeline (all steps sequentially)
py scripts/advanced_features/agentic_runner.py
```

---

## 🤝 Grounded RAG Search Usage

The **Grounded RAG Search** tab in the dashboard lets you query project documentation in plain English:

| Query | What you get |
|-------|-------------|
| `delinquency` | Field definition for `delinquency_status` + the delinquency progression validation rule |
| `current_upb` | Field definition for `current_upb` + the balance consistency rule |
| `zero_balance_code` | Disposition reason code definitions |
| `balance consistency` | The balance consistency inequality check rule |
| `ltv` | Loan-to-Value ratio field definition |

> All results are grounded in `data_dictionary.md` and `validation_rules.json`. No LLM hallucinations.

---

## 📝 License

For educational and competition purposes only. Data sourced from the Intain Campus FinTech Challenge 2026 dataset.
