"""
=================================================================
TASK 2: PREPAYMENT MODELING PIPELINE
Loan Performance Intelligence Engine
Intain Campus FinTech Challenge 2026

Trains Logistic Regression baseline and XGBoost model on the
remediated, leakage-free chronological datasets.
Outputs predictions, submission.csv, SHAP explainers, and model card.
=================================================================
"""
import pandas as pd
import numpy as np
import os
import sys
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, brier_score_loss, confusion_matrix
import xgboost as xgb
import shap

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# PATHS
DATA_DIR    = "e:/intain/data_final"
OUTPUT_DIR  = "e:/intain/data_final/outputs"
LOGS_DIR    = "e:/intain/data_final/logs"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# ---------------------------------------------------------------
# STEP 1: LOAD DATA & PREPROCESS
# ---------------------------------------------------------------
print("=" * 70)
print("STEP 1: LOADING DATA & SCHEMA VERIFICATION")
print("=" * 70)

train_df = pd.read_csv(os.path.join(DATA_DIR, "train_final.csv"))
test_df  = pd.read_csv(os.path.join(DATA_DIR, "test_final.csv"))

print(f"  Train data loaded: {train_df.shape}")
print(f"  Test data loaded:  {test_df.shape}")

# Parse dates
train_df['reporting_date'] = pd.to_datetime(train_df['reporting_date'])
test_df['reporting_date']  = pd.to_datetime(test_df['reporting_date'])

# Targets
target_col = 'next_12m_prepayment_flag'
print(f"  Target variable: {target_col}")
print(f"  Target positive rate: {train_df[target_col].mean()*100:.3f}%")

# Features classification
numeric_features = [
    'delinquency_status_lag1', 'current_upb_lag1', 'current_interest_rate_lag1',
    'loan_age', 'remaining_months', 'orig_upb', 'credit_score', 'ltv', 'dti',
    'upb_pct_of_orig', 'term_pct_elapsed'
]
categorical_features = ['state', 'loan_purpose', 'property_type', 'vintage']

print(f"  Numeric features ({len(numeric_features)}): {numeric_features}")
print(f"  Categorical features ({len(categorical_features)}): {categorical_features}")

# ---------------------------------------------------------------
# STEP 2: TIME-AWARE SPLIT (CHRONOLOGICAL)
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: CHRONOLOGICAL TIME-AWARE SPLITTING")
print("=" * 70)

# Chronological split boundary: 
# Train: Jan 2025 - Sep 2025 (< 2025-10-01)
# Validation: Oct 2025 - Nov 2025 (>= 2025-10-01)
# Test: Dec 2025 (test_df)

split_date = '2025-10-01'
train_split = train_df[train_df['reporting_date'] < split_date].copy()
val_split   = train_df[train_df['reporting_date'] >= split_date].copy()

print(f"  Train split (Jan - Sep 2025): {train_split.shape[0]:,} rows")
print(f"  Val split (Oct - Nov 2025):   {val_split.shape[0]:,} rows")
print(f"  Test split (Dec 2025):         {test_df.shape[0]:,} rows")

X_train = train_split[numeric_features + categorical_features]
y_train = train_split[target_col]

X_val = val_split[numeric_features + categorical_features]
y_val = val_split[target_col]

X_test = test_df[numeric_features + categorical_features]

# Preprocessing Pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Fit and transform
print("\n  Fitting preprocessing pipeline...")
X_train_proc = preprocessor.fit_transform(X_train)
X_val_proc   = preprocessor.transform(X_val)
X_test_proc  = preprocessor.transform(X_test)

# Get feature names after onehot
cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
cat_feature_names = list(cat_encoder.get_feature_names_out(categorical_features))
feature_names = numeric_features + cat_feature_names
print(f"  Processed features shape: {X_train_proc.shape[1]} features")

# ---------------------------------------------------------------
# STEP 3: BASELINE MODEL (LOGISTIC REGRESSION)
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: BASELINE MODEL (LOGISTIC REGRESSION)")
print("=" * 70)

lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr_model.fit(X_train_proc, y_train)

# Evaluate on Train & Val
y_train_pred_lr = lr_model.predict_proba(X_train_proc)[:, 1]
y_val_pred_lr   = lr_model.predict_proba(X_val_proc)[:, 1]

lr_roc_val = roc_auc_score(y_val, y_val_pred_lr)
lr_pr_prec, lr_pr_rec, _ = precision_recall_curve(y_val, y_val_pred_lr)
lr_pr_auc_val = auc(lr_pr_rec, lr_pr_prec)
lr_brier_val = brier_score_loss(y_val, y_val_pred_lr)

print(f"  [Logistic Regression Validation Results]")
print(f"    ROC-AUC:  {lr_roc_val:.4f}")
print(f"    PR-AUC:   {lr_pr_auc_val:.4f}")
print(f"    Brier:    {lr_brier_val:.4f}")

# ---------------------------------------------------------------
# STEP 4: IMPROVED MODEL (XGBOOST)
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4: IMPROVED MODEL (XGBOOST)")
print("=" * 70)

# Calculate scale_pos_weight
imbalance_ratio = (len(y_train) - sum(y_train)) / sum(y_train)
print(f"  Imbalance ratio (neg:pos): {imbalance_ratio:.2f}")

xgb_model = xgb.XGBClassifier(
    scale_pos_weight=imbalance_ratio,
    n_estimators=100,
    max_depth=6,
    learning_rate=0.05,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train_proc, y_train)

# Predict
y_train_pred_xgb = xgb_model.predict_proba(X_train_proc)[:, 1]
y_val_pred_xgb   = xgb_model.predict_proba(X_val_proc)[:, 1]

xgb_roc_val = roc_auc_score(y_val, y_val_pred_xgb)
xgb_pr_prec, xgb_pr_rec, xgb_thresholds = precision_recall_curve(y_val, y_val_pred_xgb)
xgb_pr_auc_val = auc(xgb_pr_rec, xgb_pr_prec)
xgb_brier_val = brier_score_loss(y_val, y_val_pred_xgb)

print(f"  [XGBoost Validation Results]")
print(f"    ROC-AUC:  {xgb_roc_val:.4f}")
print(f"    PR-AUC:   {xgb_pr_auc_val:.4f}")
print(f"    Brier:    {xgb_brier_val:.4f}")

# ---------------------------------------------------------------
# STEP 5: PROBABILITY CALIBRATION
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 5: PROBABILITY CALIBRATION (PLATT SCALING)")
print("=" * 70)

# We train a calibration model on validation set predictions (Platt Scaling)
# Ensure we do not leak this calibration back, but it's evaluated on validation predictions.
# Since test data is out-of-sample, we'll use this Platt scaling model for final test predictions.
val_preds_df = pd.DataFrame({'raw_prob': y_val_pred_xgb, 'label': y_val})
calibrator = LogisticRegression()
calibrator.fit(val_preds_df[['raw_prob']], val_preds_df['label'])

# Apply calibration
y_val_calibrated = calibrator.predict_proba(val_preds_df[['raw_prob']])[:, 1]
cal_brier_val = brier_score_loss(y_val, y_val_calibrated)
cal_roc_val = roc_auc_score(y_val, y_val_calibrated)
cal_pr_prec, cal_pr_rec, _ = precision_recall_curve(y_val, y_val_calibrated)
cal_pr_auc_val = auc(cal_pr_rec, cal_pr_prec)

print(f"  Before Calibration (XGBoost Raw): Brier = {xgb_brier_val:.4f}")
print(f"  After Calibration (Platt Scaled): Brier = {cal_brier_val:.4f}")
print(f"  Calibrated Validation ROC-AUC:    {cal_roc_val:.4f}")
print(f"  Calibrated Validation PR-AUC:     {cal_pr_auc_val:.4f}")

# ---------------------------------------------------------------
# STEP 6: OPTIMAL THRESHOLD SELECTION
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 6: OPTIMAL THRESHOLD SELECTION")
print("=" * 70)

# Find the threshold that maximizes validation F1 score
best_f1 = 0
best_thresh = 0.5

# We use calibrated probabilities to find the threshold
for thresh in np.linspace(0.01, 0.99, 100):
    y_pred_bin = (y_val_calibrated >= thresh).astype(int)
    f1 = f1_score(y_val, y_pred_bin)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh

print(f"  Optimal decision threshold (maximizes F1): {best_thresh:.4f}")
print(f"  Best F1 Score achieved: {best_f1:.4f}")

# Metrics at optimal threshold
y_val_pred_bin = (y_val_calibrated >= best_thresh).astype(int)
conf_matrix = confusion_matrix(y_val, y_val_pred_bin)
tn, fp, fn, tp = conf_matrix.ravel()
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0

print(f"    Confusion Matrix:")
print(f"      TN: {tn:,}  FP: {fp:,}")
print(f"      FN: {fn:,}  TP: {tp:,}")
print(f"    Precision: {precision:.4f}")
print(f"    Recall:    {recall:.4f}")

# Recall at fixed precision (e.g. at 20% precision)
recall_at_20_prec = 0
for p, r in zip(cal_pr_prec, cal_pr_rec):
    if p >= 0.20:
        recall_at_20_prec = max(recall_at_20_prec, r)
print(f"    Recall at 20% Precision constraint: {recall_at_20_prec:.4f}")

# ---------------------------------------------------------------
# STEP 7: EXPLAINABILITY (SHAP GLOBAL AND LOCAL)
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 7: MODEL EXPLAINABILITY")
print("=" * 70)

# Try SHAP first, with fallback to native XGBoost importances
shap_values = None
explainer = None
try:
    print("  Attempting to initialize SHAP TreeExplainer on booster...")
    # Use get_booster() to avoid scikit-learn wrapper base_score string formatting bug in shap
    explainer = shap.TreeExplainer(xgb_model.get_booster())
    shap_sample = X_val_proc[np.random.choice(X_val_proc.shape[0], 1000, replace=False)]
    shap_values = explainer.shap_values(shap_sample)
    
    # Global importance
    mean_shap = np.abs(shap_values).mean(axis=0)
    global_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_shap
    }).sort_values(by='importance', ascending=False)

    print("  Top 10 Global Features (SHAP Mean Absolute Value):")
    print(global_importance.head(10).to_string(index=False))
except Exception as e:
    print(f"  SHAP explainer failed: {e}")
    print("  Falling back to XGBoost native feature importances...")
    # Fallback to XGBoost native importances
    importances = xgb_model.feature_importances_
    global_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)
    
    print("  Top 10 Global Features (XGBoost Feature Importance):")
    print(global_importance.head(10).to_string(index=False))

# ---------------------------------------------------------------
# STEP 8: ANOMALY DETECTION FOR SUBMISSION
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 8: ANOMALY DETECTION FOR SUBMISSION")
print("=" * 70)

# Isolation Forest to compute anomaly scores on the features
iso_forest = IsolationForest(contamination=0.01, random_state=42)
iso_forest.fit(X_train_proc)

train_anomaly_scores = iso_forest.score_samples(X_train_proc)
test_anomaly_scores  = iso_forest.score_samples(X_test_proc)

min_score = min(train_anomaly_scores.min(), test_anomaly_scores.min())
max_score = max(train_anomaly_scores.max(), test_anomaly_scores.max())
test_scaled_anom_score = 1.0 - (test_anomaly_scores - min_score) / (max_score - min_score + 1e-9)

print(f"  Anomaly scores computed. Mean test anomaly score: {test_scaled_anom_score.mean():.4f}")

# ---------------------------------------------------------------
# STEP 9: SUBMISSION EXPORT
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 9: EXPORT SUBMISSION AND MODEL ARTIFACTS")
print("=" * 70)

# Predict on test
y_test_pred_raw = xgb_model.predict_proba(X_test_proc)[:, 1]
y_test_pred_cal = calibrator.predict_proba(y_test_pred_raw.reshape(-1, 1))[:, 1]
y_test_pred_bin = (y_test_pred_cal >= best_thresh).astype(int)

# Create submission dataframe matching template:
sub = pd.DataFrame()
sub['loan_id']          = test_df['loan_id']
sub['reporting_period'] = test_df['reporting_period']
sub['prob_default']     = 0.0 # Non-modelable target due to 0 positives
sub['prob_prepay']      = np.round(y_test_pred_cal, 4)
sub['next_state']       = np.where(y_test_pred_bin == 1, 'Prepaid', 'Current')
sub['exception_type']   = 'None'
sub['anomaly_score']    = np.round(test_scaled_anom_score, 4)

# Top drivers for each prediction using SHAP or global importances fallback
top_drivers_list = []
if explainer is not None and shap_values is not None:
    try:
        test_shap_values = explainer.shap_values(X_test_proc)
        for i in range(len(test_df)):
            local_s = test_shap_values[i]
            top_indices = np.argsort(np.abs(local_s))[-2:][::-1]
            drivers = [feature_names[idx] for idx in top_indices]
            drivers_clean = [d.split('_')[0] for d in drivers]
            top_drivers_list.append(f"{drivers_clean[0]}, {drivers_clean[1]}")
    except Exception as e:
        print(f"  Failed local SHAP prediction: {e}. Falling back to global drivers.")
        explainer = None

if explainer is None or not top_drivers_list:
    # Use top 2 global features as fallback
    top_global = global_importance.head(2)['feature'].tolist()
    top_global_clean = [g.split('_')[0] for g in top_global]
    fallback_drivers = f"{top_global_clean[0]}, {top_global_clean[1]}"
    top_drivers_list = [fallback_drivers] * len(test_df)

sub['top_drivers']      = top_drivers_list
sub['action']           = np.where(y_test_pred_cal >= best_thresh, 'Flag_Prepay_Risk', 'Accept')
sub['confidence']       = np.round(np.where(y_test_pred_bin == 1, y_test_pred_cal, 1.0 - y_test_pred_cal), 4)

# Save submission.csv to target folders
sub.to_csv("e:/intain/data_final/submission.csv", index=False)
sub.to_csv("e:/intain/data/submission.csv", index=False)
print("  Submission.csv saved successfully to data_final/ and data/ directories.")

# Save artifacts
joblib.dump(xgb_model, os.path.join(OUTPUT_DIR, "xgboost_prepay_model.pkl"))
joblib.dump(preprocessor, os.path.join(OUTPUT_DIR, "preprocessor.pkl"))
joblib.dump(calibrator, os.path.join(OUTPUT_DIR, "platt_calibrator.pkl"))
joblib.dump(iso_forest, os.path.join(OUTPUT_DIR, "anomaly_detector.pkl"))
print("  All model artifacts saved to data_final/outputs/.")

# ---------------------------------------------------------------
# STEP 10: GENERATE MODEL CARD AND LOG
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 10: MODEL CARD & LOG GENERATION")
print("=" * 70)

# Save experiment log
with open(os.path.join(LOGS_DIR, "experiment_log.txt"), "w") as f:
    f.write("EXPERIMENT LOG\n")
    f.write("=" * 40 + "\n")
    f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Baseline Model: Logistic Regression (balanced)\n")
    f.write(f"  Val ROC-AUC: {lr_roc_val:.4f}\n")
    f.write(f"  Val PR-AUC:  {lr_pr_auc_val:.4f}\n")
    f.write(f"  Val Brier:   {lr_brier_val:.4f}\n\n")
    f.write(f"Improved Model: XGBoost\n")
    f.write(f"  Val ROC-AUC: {xgb_roc_val:.4f}\n")
    f.write(f"  Val PR-AUC:  {xgb_pr_auc_val:.4f}\n")
    f.write(f"  Val Brier (raw): {xgb_brier_val:.4f}\n")
    f.write(f"  Val Brier (calibrated): {cal_brier_val:.4f}\n\n")
    f.write(f"Optimal Threshold: {best_thresh:.4f}\n")
    f.write(f"  F1 Score:   {best_f1:.4f}\n")
    f.write(f"  Precision:  {precision:.4f}\n")
    f.write(f"  Recall:     {recall:.4f}\n")
    f.write(f"  Recall at 20% precision: {recall_at_20_prec:.4f}\n")

# Model Card
model_card = f"""=================================================================
MODEL CARD: LOAN PERFORMANCE PREPAYMENT ENGINE
=================================================================

1. Objective:
   Predict the probability that a single-family mortgage will prepay 
   (UPB reaches 0) within the next 12 months (next_12m_prepayment_flag).

2. Target Definition:
   - next_12m_prepayment_flag: 1 if loan terminates with zero UPB in 
     reporting months t+1 to t+12; 0 otherwise.

3. Features Used:
   - Lagged Monthly Performance (t-1): current_upb_lag1, delinquency_status_lag1, current_interest_rate_lag1
   - Current Temporal: loan_age, remaining_months
   - Static Origination: orig_upb, credit_score, ltv, dti, state, loan_purpose, property_type, vintage
   - Derived: upb_pct_of_orig, term_pct_elapsed

4. Validation Scheme:
   - Time-Aware split to respect panel chronology:
     - Train: Jan 2025 - Sep 2025 (374,275 rows)
     - Validation: Oct 2025 - Nov 2025 (93,395 rows)
     - Test (Holdout): Dec 2025 (32,176 rows)

5. Leakage Controls:
   - Same-row delinquency_status and current_upb are strictly LAGGED by 1 month.
   - zero_balance_code is completely EXCLUDED (post-event indicator).
   - Training set filtered to dates strictly before the test period to prevent future-data overlap.

6. Performance Metrics (Validation Set):
   - Baseline (Logistic Regression):
     * ROC-AUC:  {lr_roc_val:.4f}
     * PR-AUC:   {lr_pr_auc_val:.4f}
     * Brier:    {lr_brier_val:.4f}
   - Improved Model (XGBoost Calibrated):
     * ROC-AUC:  {cal_roc_val:.4f}
     * PR-AUC:   {cal_pr_auc_val:.4f}
     * Brier:    {cal_brier_val:.4f}
     * F1-Score: {best_f1:.4f} (at threshold {best_thresh:.4f})
     * Precision: {precision:.4f}
     * Recall:    {recall:.4f}

7. Limitations:
   - Delinquency/Default targets are non-viable (0 positive examples in vintage).
   - Servicer updates file excluded from modeling due to 100% null delinquency and same-row leakage.
"""

with open(os.path.join(LOGS_DIR, "model_card.txt"), "w") as f:
    f.write(model_card)

print("  Model Card saved successfully.")
print("\nModeling pipeline run finished.")
