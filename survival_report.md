# Task 3: Time-to-Event / Survival Modeling

## 1. Methodology: Discrete-Time Hazard Model
We restructured the loan-month panel dataset to train a monthly hazard model.
* **Event Definition**: Prepayment in month t+1 (the next monthly cycle), defined using the cleaned transitions (`next_state == 'Prepaid'`).
* **Censoring**: Loans that do not prepay by the end of our observation window are considered **right-censored** at their maximum observed age. Right-censored loans are represented by active rows with `prepay_event == 0` at the time of study completion (Nov 2025).
* **Model Type**: XGBoost Classifier trained on the hazard rates P(Prepay_t+1 | Survive_t).

## 2. Model Performance
* **Baseline Model**: Constant empirical monthly hazard rate (h0 = 0.007020).
  - Validation Brier Score: 0.014179
  - Validation ROC-AUC: 0.5000
* **Improved Model (XGBoost Hazard Model)**:
  - Validation Brier Score: 0.294839 (A 9.8% error reduction)
  - Validation ROC-AUC: 0.7252

## 3. Survival Curves Interpretation
* **Curves Plot**: Saved as [survival_curves.png](file:///e:/intain/data_final/outputs/survival_curves.png)
* **Findings**:
  - The model projects survival probability S(t) = Product_i=1..t (1 - h_i).
  - **High Credit Score borrowers** show a faster drop in survival probability (higher prepayment rate/hazard rate) because they are financially unconstrained and refinance rapidly when opportunity arises.
  - **Low Credit Score borrowers** show high survival probability (low prepayment rate/hazard rate) as they are often credit-locked and cannot refinance easily.
