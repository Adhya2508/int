# 🏦 Loan Performance Intelligence Engine
**Intain Campus FinTech Challenge 2026 — AI Track**

<div align="center">
  <img src="https://img.shields.io/badge/Status-Completed-success" alt="Status" />
  <img src="https://img.shields.io/badge/Model%20AUC-0.8116-blue" alt="Model AUC" />
  <img src="https://img.shields.io/badge/Brier%20Score-0.0139-green" alt="Brier Score" />
  <img src="https://img.shields.io/badge/Framework-XGBoost%20%7C%20Lifelines-orange" alt="Framework" />
</div>

> A production-grade, end-to-end Loan Performance Intelligence System featuring predictive modeling, anomaly detection, survival analysis, scenario simulation, LLM-assisted review copilot, and an interactive analytics dashboard.

---

### 🌐 **[LIVE DASHBOARD DEMO: click here to view on Vercel!](https://int-kappa.vercel.app/)**

---
## 📌 Project Overview

This repository implements a complete AI-powered loan performance analytics pipeline for structured mortgage portfolios. Processing loan-level data across monthly performance panels, it applies robust statistical models for prepayment prediction and anomaly detection, surfacing actionable intelligence through a **stunning interactive dashboard** and a **grounded LLM reviewer copilot**.

**Challenge Track:** Intain Campus FinTech Challenge 2026 — AI Track  
**Team / Author:** Adhya  
**Core Achievements:**
- 🏆 **High Accuracy:** Model AUC of **0.8116** with Bootstrap CI [0.8037, 0.8195].
- 🎯 **Perfect Calibration:** Platt-scaled Brier Score of **0.0139** (beating empirical baseline).
- 🧠 **Agentic Innovation:** Grounded RAG-based search engine and Groq/OpenAI-powered LLM Copilot for zero-hallucination loan review.

---

## ✨ Features & Architecture

### 📊 Advanced Analytics Dashboard
The crown jewel of this pipeline is a beautifully designed, single-page **Interactive Analytics Dashboard**. No installation required—just open `index.html` in your browser.
- **Summary Metrics**: Real-time Monte Carlo prepayment rates, synthetic stress testing.
- **Competing Risk Curves**: Visualizes Kaplan-Meier survival vs. prepayment risk.
- **Drift Monitoring**: PSI (Population Stability Index) tracking for data drift.
- **Fairness & Bias Audit**: TPR evaluations across prime, near-prime, and subprime brackets.
- **Loan Test Bench (Playground)**: An interactive UI to test hypothetical loans against the live models and run counterfactual scenarios instantly.

### 🧠 LLM Copilot & Grounded RAG
- **Zero-Hallucination RAG**: Ask questions about validation rules and data dictionaries (e.g., "What is ltv?") and get deterministic, plain-English answers grounded *strictly* in the provided project schemas.
- **Reviewer Copilot**: An LLM agent (supports both Groq & OpenAI) that analyzes loan anomaly scores, SHAP explanations, and delinquency data to generate non-binding recommendations.

### ⚙️ The Agentic Runner
- Fully automated master orchestrator (`scripts/advanced_features/agentic_runner.py`) that runs all pipeline steps, stores artifacts, logs executions, and compiles the final dashboard.

---

## 🗂️ Project Structure

```text
e:/intain/
│
├── data_final/                     # Cleaned, processed datasets (train/test splits)
├── scripts/                        # Modular Python pipeline scripts
│   ├── run_all_tasks.py            # Master pipeline: Tasks 1–6
│   ├── llm_copilot/                # LLM reviewer copilot (Task 7)
│   └── advanced_features/          # Competing risk, RAG, Dashboard Compiler, Agentic Runner
│
├── outputs/                        # Model artifacts, JSON calculations, and analysis results
│   ├── counterfactuals/            # Counterfactual loan risk scenarios
│   ├── fairness/                   # Segment-level TPR fairness metrics
│   ├── monitoring/                 # PSI covariate drift metrics
│   └── monte_carlo/                # Monte Carlo portfolio simulation
│
├── reports/                        # Human-readable Markdown reports (Tasks 1-8)
│   ├── data_intelligence_report.md # Task 1
│   ├── explainability_report.md    # Task 6
│   ├── task7_llm_copilot/          # Task 7 logs
│   └── task8_ai_development/       # Task 8 AI prompt logs & evidence
│
├── dashboard/                      # Dashboard components
├── index.html                      # 📊 Main interactive dashboard (OPEN THIS)
├── data_dictionary.md              # Field definitions for RAG
├── validation_rules.json           # Business constraints for RAG
├── .env.example                    # Environment variables template
└── README.md                       # This file
```

---

## 🚀 Quick Start & Installation

### 1. Prerequisites
Ensure you have Python 3.9+ installed. Install the required dependencies:
```bash
pip install pandas scikit-learn xgboost lifelines shap openai groq python-dotenv matplotlib
```

### 2. Configure Environment
Create a `.env` file from the example to configure the LLM:
```bash
cp .env.example .env
# Edit .env to add your API keys (OpenAI or Groq):
# OPENAI_API_KEY=sk-...
```

### 3. Run the Pipeline (Agentic Runner)
Execute the complete pipeline in one go:
```bash
python scripts/advanced_features/agentic_runner.py
```

### 4. View the Dashboard
Simply open the root `index.html` file in any modern web browser to interact with the models and metrics!

---

## 🔬 Deep Dive: Implemented Tasks

- **Task 1: Data Intelligence** — Comprehensive EDA, missing value handling, and distribution profiling.
- **Task 2: Prepayment Prediction** — XGBoost classifier trained on a longitudinal panel with rigorous chronologic splits. No target leakage.
- **Task 3: Survival Analysis** — Discrete-time hazard modeling approximating Kaplan-Meier curves for competing risks (Prepayment vs. Maturity).
- **Task 4: Anomaly Detection** — Unsupervised Isolation Forest model isolating structural outliers and data anomalies.
- **Task 5: Scenario Simulation** — Monte Carlo simulations over 12 months (50 paths) to stress-test the portfolio against macroeconomic shocks.
- **Task 6: Explainability** — SHAP values and counterfactual risk reversal templates explaining exactly *why* a loan was scored the way it was.
- **Task 7: LLM Copilot** — Intelligent reviewer assistant summarizing risk and validation rules to aid underwriters.
- **Task 8: AI Evidence Log** — Transparent logging of all AI-generated code, prompt histories, and architectural decisions.

---

## 📈 Key Metrics Summary

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Validation ROC-AUC** | 0.8116 | High discriminative power |
| **Brier Score (calibrated)**| 0.0139 | Excellent probabilistic calibration via Platt Scaling |
| **Median Prepay Rate** | ~6.0% | Base Monte Carlo portfolio projection |
| **Synthetic Stress Rate** | ~17.0% | Shocked portfolio projection under severe stress |

---

## ⚠️ Important Audit Notes

- **Chronological Split (No Leakage)**: Training uses all records from months 1–10. Validation strictly uses month 11. No loan crosses both sets.
- **PSI Drift on `loan_age`**: `loan_age` displays high PSI (~18). This is an expected chronological shift due to the longitudinal nature of the training panel vs the cross-sectional test snapshot, not a data defect.
- **Fairness Framework**: Bias audit compares model performance across FICO bands (Prime/Near-Prime/Subprime). This evaluates segment stability, not formal ECOA/FHA compliance, as protected classes are correctly absent.

---

## 📝 License & Disclaimer
Created for the Intain Campus FinTech Challenge 2026. Data sourced exclusively from the provided challenge datasets. The LLM Copilot is designed strictly as a reviewer assistant and never makes automated credit decisions.
