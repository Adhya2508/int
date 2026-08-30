"""
=================================================================
FINAL DATA PASS & AUDIT  --  Task 1 -> Task 2 Handoff
Loan Performance Intelligence Engine
Intain Campus FinTech Challenge 2026

This script:
  1. Filters train data to remove future dates (removes temporal leakage)
  2. Tightens the feature set and adds derived features
  3. Handles first-row lag nulls safely
  4. Drops non-viable targets (keeps only viable ones)
  5. Saves finalized datasets
  6. Runs a strict post-fix audit on finalized files
  7. Outputs all logs and handoff specifications
=================================================================
"""
import pandas as pd
import numpy as np
import json
import os
import sys
from collections import OrderedDict

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# PATHS
PREV_CLEAN  = "e:/intain/data_cleaned"
FINAL_DIR   = "e:/intain/data_final"
FINAL_LOGS  = "e:/intain/data_final/logs"
FINAL_QUAR  = "e:/intain/data_final/quarantine"
RULES_FILE  = "e:/intain/validation_rules.json"

for d in [FINAL_DIR, FINAL_LOGS, FINAL_QUAR]:
    os.makedirs(d, exist_ok=True)

quarantine_log = []
fix_log        = []
audit_issues   = []

def q_log(src, lid, rp, reason):
    quarantine_log.append({"source": src, "loan_id": lid, "reporting_period": rp, "reason": reason})

def f_log(task, desc, n):
    fix_log.append({"task": task, "description": desc, "rows_affected": n})

def a_issue(sev, fname, desc, detail=""):
    audit_issues.append((sev, fname, desc, detail))

# ---------------------------------------------------------------
# LOAD & FILTER (Fix temporal leakage at load time)
# ---------------------------------------------------------------
print("=" * 70)
print("LOADING & FILTERING DATA")
print("=" * 70)

train_raw = pd.read_csv(os.path.join(PREV_CLEAN, "train_modeling_ready.csv"))
test_raw  = pd.read_csv(os.path.join(PREV_CLEAN, "test_modeling_ready.csv"))
static = pd.read_csv(os.path.join(PREV_CLEAN, "static_cleaned.csv"))

serv_path = os.path.join(PREV_CLEAN, "servicer_updates_cleaned.csv")
serv = pd.read_csv(serv_path) if os.path.exists(serv_path) else None

with open(RULES_FILE) as f:
    rules = json.load(f)

# Cast dates
train_raw['reporting_date'] = pd.to_datetime(train_raw['reporting_date'])
test_raw['reporting_date']  = pd.to_datetime(test_raw['reporting_date'])

# FILTER OUT FUTURE TRAINING DATA (Dec 2025 and later)
# Test is strictly 2025-12-01. Any train record >= 2025-12-01 is future data.
n_before = len(train_raw)
train = train_raw[train_raw['reporting_date'] < '2025-12-01'].copy()
n_after = len(train)
n_filtered = n_before - n_after

print(f"  Raw train rows: {n_before:,}")
print(f"  Filtered train rows (reporting_date < 2025-12-01): {n_after:,}")
print(f"  Removed future/overlapping rows: {n_filtered:,}")
f_log("C0", "Filtered out train rows with reporting_date >= 2025-12-01 to fix temporal leakage", n_filtered)

test = test_raw.copy()
print(f"  Test rows: {len(test):,}")
print(f"  Static rows: {len(static):,}")
print()

# ---------------------------------------------------------------
# STEP 2: TIGHTEN FEATURE SET & REMAINING FIXES
# ---------------------------------------------------------------
print("=" * 70)
print("STEP 2: TIGHTEN FEATURE SET & REMAINING FIXES")
print("=" * 70)

# Drop redundant columns
for df in [train, test]:
    if 'reporting_period_str' in df.columns:
        df.drop(columns=['reporting_period_str'], inplace=True)

# Add derived features
print("  Adding derived features...")
for name, df in [("train", train), ("test", test)]:
    df['upb_pct_of_orig'] = np.where(
        (df['orig_upb'] > 0) & (df['current_upb_lag1'].notna()),
        df['current_upb_lag1'] / df['orig_upb'],
        np.nan
    )
    df['term_pct_elapsed'] = np.where(
        (df['loan_age'] + df['remaining_months']) > 0,
        df['loan_age'] / (df['loan_age'] + df['remaining_months']),
        np.nan
    )

f_log("4b", "Added derived features: upb_pct_of_orig, term_pct_elapsed", len(train) + len(test))

# Impute lag nulls on first row of loan panel safely
print("  Handling first-row lag nulls...")
train_lag_null = train['delinquency_status_lag1'].isna()
n_lag_null = train_lag_null.sum()
train.loc[train_lag_null, 'delinquency_status_lag1'] = 0.0
train.loc[train_lag_null & train['current_upb_lag1'].isna(), 'current_upb_lag1'] = \
    train.loc[train_lag_null & train['current_upb_lag1'].isna(), 'orig_upb']
train.loc[train_lag_null & train['current_interest_rate_lag1'].isna(), 'current_interest_rate_lag1'] = \
    train.groupby('loan_id')['current_interest_rate_lag1'].transform(lambda x: x.bfill())

# Recompute derived for imputed
train.loc[train_lag_null, 'upb_pct_of_orig'] = np.where(
    train.loc[train_lag_null, 'orig_upb'] > 0,
    train.loc[train_lag_null, 'current_upb_lag1'] / train.loc[train_lag_null, 'orig_upb'],
    np.nan
)
print(f"    Imputed {n_lag_null:,} lag nulls in train")

test_lag_null = test['delinquency_status_lag1'].isna()
n_test_lag_null = test_lag_null.sum()
if n_test_lag_null > 0:
    test.loc[test_lag_null, 'delinquency_status_lag1'] = 0.0
    test.loc[test_lag_null & test['current_upb_lag1'].isna(), 'current_upb_lag1'] = \
        test.loc[test_lag_null & test['current_upb_lag1'].isna(), 'orig_upb']
    print(f"    Imputed {n_test_lag_null:,} lag nulls in test")

# Clip negative loan_age
for name, df in [("train", train), ("test", test)]:
    neg_age_mask = df['loan_age'] < 0
    n_neg = neg_age_mask.sum()
    if n_neg > 0:
        df.loc[neg_age_mask, 'loan_age'] = 0.0
        print(f"    Clipped {n_neg} negative loan_age to 0 in {name}")

# Exclude non-viable targets (zero positives) from modeling files
non_viable_targets = ['next_1m_default_flag', 'next_3m_default_flag', 'next_6m_default_flag',
                      'next_12m_default_flag', 'next_3m_delinquency_flag']
train.drop(columns=[c for c in non_viable_targets if c in train.columns], inplace=True)

# Boundary target rows for next_state
if 'next_state' in train.columns:
    train['next_state'] = train['next_state'].replace('nan', pd.NA).fillna('Unknown')

print()

# ---------------------------------------------------------------
# STEP 3: SERVICER FILE FINAL DECISION
# ---------------------------------------------------------------
print("=" * 70)
print("STEP 3: SERVICER FILE FINAL DECISION")
print("=" * 70)

servicer_decision = "EXCLUDED_FROM_MODELING"
print(f"  Servicer decision: {servicer_decision}")
print("  Rationale:")
print("    1. servicer delinquency_status is 100% null in raw data -- unusable for default validation.")
print("    2. servicer current_upb represents same-period data -- including it as a feature would")
print("       reintroduce same-period target leakage.")
print("    3. update_date has zero temporal variance (single mock value 2025-04-01).")
print("  Action: Exclude servicerupdates entirely from modeling. Retain in logs for audit.")
print()

# ---------------------------------------------------------------
# STEP 4: REBUILD DEFINITIVE MODELING DATASETS
# ---------------------------------------------------------------
print("=" * 70)
print("STEP 4: REBUILD DEFINITIVE MODELING DATASETS")
print("=" * 70)

final_features = [
    'loan_id', 'reporting_period', 'reporting_date',
    'delinquency_status_lag1', 'current_upb_lag1', 'current_interest_rate_lag1',
    'loan_age', 'remaining_months',
    'orig_upb', 'credit_score', 'ltv', 'dti',
    'state', 'loan_purpose', 'property_type', 'vintage',
    'upb_pct_of_orig', 'term_pct_elapsed',
]
final_targets = ['next_12m_prepayment_flag', 'next_state']

train_final = train[[c for c in final_features + final_targets if c in train.columns]].copy()
test_final  = test[[c for c in final_features if c in test.columns]].copy()

# Save final clean data
train_final.to_csv(os.path.join(FINAL_DIR, "train_final.csv"), index=False)
test_final.to_csv(os.path.join(FINAL_DIR, "test_final.csv"), index=False)
static.to_csv(os.path.join(FINAL_DIR, "static_final.csv"), index=False)

print(f"  Saved: train_final.csv  ({len(train_final):,} rows)")
print(f"  Saved: test_final.csv   ({len(test_final):,} rows)")
print(f"  Saved: static_final.csv ({len(static):,} rows)")
print()

# ---------------------------------------------------------------
# STEP 5: STRICT POST-FIX AUDIT
# ---------------------------------------------------------------
print("=" * 70)
print("STEP 5: STRICT POST-FIX AUDIT")
print("=" * 70)

# Re-audit train_final and test_final
# 1. Duplicates
for name, df in [("train_final", train_final), ("test_final", test_final)]:
    dups = df.duplicated(subset=['loan_id', 'reporting_period']).sum()
    print(f"  {name} duplicate keys: {dups}")
    if dups > 0:
        a_issue("CRITICAL", name, f"{dups} duplicate loan-month keys")

# 2. Missing values in important features
print("\n  --- Nulls in features ---")
for name, df in [("train_final", train_final), ("test_final", test_final)]:
    for col in final_features:
        if col in df.columns:
            n_null = df[col].isna().sum()
            if n_null > 0:
                pct = n_null / len(df) * 100
                print(f"    {name}.{col}: {n_null:,} nulls ({pct:.2f}%)")
                # credit_score can have minor nulls (0.17%), others shouldn't
                if pct > 5 and col != 'credit_score':
                    a_issue("HIGH", name, f"Column '{col}' is {pct:.1f}% null")

# 3. Invalid dates or broken date ordering
for name, df in [("train_final", train_final), ("test_final", test_final)]:
    df['reporting_date'] = pd.to_datetime(df['reporting_date'])
    bad_dates = df['reporting_date'].isna().sum()
    if bad_dates > 0:
        a_issue("CRITICAL", name, f"{bad_dates} unparseable reporting_dates")
        
    sorted_df = df.sort_values(['loan_id', 'reporting_date'])
    sorted_df['prev_date'] = sorted_df.groupby('loan_id')['reporting_date'].shift(1)
    ordering_breaks = sorted_df[(sorted_df['prev_date'].notna()) & 
                                (sorted_df['reporting_date'] <= sorted_df['prev_date'])]
    if len(ordering_breaks) > 0:
        a_issue("HIGH", name, f"{len(ordering_breaks)} date ordering breaks")

# 4. Temporal leakage (train date >= test date)
train_max_date = train_final['reporting_date'].max()
test_min_date  = test_final['reporting_date'].min()
print(f"\n  Train max reporting date: {train_max_date}")
print(f"  Test  min reporting date: {test_min_date}")
if test_min_date <= train_max_date:
    a_issue("CRITICAL", "train_final", f"Temporal leakage: test starts at {test_min_date} but train extends to {train_max_date}")
else:
    print("  [PASS] Test is strictly after train (no temporal leakage)")

# 5. Same-row leakage columns
unsafe_cols = ['delinquency_status', 'current_upb', 'current_interest_rate', 'zero_balance_code']
found_leak_train = [c for c in unsafe_cols if c in train_final.columns]
found_leak_test  = [c for c in unsafe_cols if c in test_final.columns]
if found_leak_train:
    a_issue("CRITICAL", "train_final", f"Leakage columns present: {found_leak_train}")
if found_leak_test:
    a_issue("CRITICAL", "test_final", f"Leakage columns present: {found_leak_test}")

# 6. Targets in test
test_targets = [c for c in final_targets if c in test_final.columns]
if test_targets:
    a_issue("CRITICAL", "test_final", f"Target columns leaked into test: {test_targets}")

# 7. Balance Consistency
bal_violations = train_final[train_final['current_upb_lag1'] > train_final['orig_upb'] * 1.05]
print(f"  Balance consistency violations (lagged UPB > orig * 1.05): {len(bal_violations)}")
if len(bal_violations) > 0:
    a_issue("HIGH", "train_final", f"{len(bal_violations)} lagged UPB violations")

# 8. Delinquency progressions
sorted_t = train_final.sort_values(['loan_id', 'reporting_date'])
sorted_t['prev_dlq'] = sorted_t.groupby('loan_id')['delinquency_status_lag1'].shift(1)
dlq_violations = sorted_t[(sorted_t['prev_dlq'].notna()) & 
                          (sorted_t['delinquency_status_lag1'] > sorted_t['prev_dlq'] + 1)]
print(f"  Delinquency progression violations: {len(dlq_violations)}")
if len(dlq_violations) > 0:
    a_issue("HIGH", "train_final", f"{len(dlq_violations)} delinquency jumps")

# 9. DQ Score
def compute_dq(row):
    checks = ['loan_age', 'remaining_months', 'delinquency_status_lag1', 'current_upb_lag1', 'orig_upb', 'credit_score', 'ltv', 'dti']
    valid = sum(1 for c in checks if pd.notna(row.get(c)))
    return valid / len(checks)

sample_dq = train_final.sample(min(50000, len(train_final)), random_state=42)
scores = sample_dq.apply(compute_dq, axis=1)
batch_score = scores.mean()
print(f"\n  Batch DQ Score: {batch_score:.4f}")
if batch_score < 0.95:
    a_issue("HIGH", "train_final", f"Batch DQ Score is {batch_score:.4f} (below 0.95)")

print()

# ---------------------------------------------------------------
# STEP 6: VERDICT & SUMMARY
# ---------------------------------------------------------------
print("=" * 70)
print("FINAL AUDIT SUMMARY")
print("=" * 70)

crit = sum(1 for i in audit_issues if i[0] == 'CRITICAL')
high = sum(1 for i in audit_issues if i[0] == 'HIGH')
med  = sum(1 for i in audit_issues if i[0] == 'MEDIUM')

print(f"  CRITICAL: {crit}")
print(f"  HIGH    : {high}")
print(f"  MEDIUM  : {med}")
print()

for idx, (sev, fname, desc, detail) in enumerate(audit_issues, 1):
    print(f"  [{sev}] #{idx} ({fname}): {desc}")

print()
print("=" * 70)
print("FINAL READY VERDICT")
print("=" * 70)

if crit > 0:
    print("  VERDICT: NOT READY FOR MODELING (Critical issues remain)")
else:
    print("  VERDICT: READY FOR MODELING (Task 2 Prep Complete)")
    print("  Primary Target: next_12m_prepayment_flag (Viable: 7.93% positive rate)")
    print("  Secondary Target: next_state (Viable: Current/Prepaid/Unknown multi-class)")
    print("  Default Targets: Deferred (0 positives in this vintage)")

# Save final logs
pd.DataFrame(quarantine_log).to_csv(os.path.join(FINAL_QUAR, "quarantined_final.csv"), index=False) if quarantine_log else None
pd.DataFrame(fix_log).to_csv(os.path.join(FINAL_LOGS, "fix_log_final.csv"), index=False)

# Write spec sheets
with open(os.path.join(FINAL_LOGS, "feature_spec.txt"), "w") as f:
    f.write("SAFE MODELING FEATURES:\n")
    for col in final_features:
        f.write(f"  - {col}\n")
        
with open(os.path.join(FINAL_LOGS, "servicer_final_decision.txt"), "w") as f:
    f.write("Servicer file excluded from modeling. delinquency_status is 100% null; current_upb is same-period leakage.\n")

with open(os.path.join(FINAL_LOGS, "target_strategy.txt"), "w") as f:
    f.write("Target strategy:\n")
    f.write("  1. next_12m_prepayment_flag (Binary classification, 7.93% positive rate, 11.6:1 imbalance ratio)\n")
    f.write("  2. next_state (Multi-class: Current, Prepaid, Unknown)\n")
    f.write("  3. Default/delinquency targets: deferred due to zero positives in this dataset vintage.\n")

print("\nProcessing complete.")
