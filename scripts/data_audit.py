"""
=================================================================
STRICT DATA-QUALITY AUDIT  –  Task 1
Loan Performance Intelligence Engine
Intain Campus FinTech Challenge 2026
=================================================================
"""
import pandas as pd
import numpy as np
import json
import sys
import os

DATA   = "e:/intain/data"
TRAIN  = os.path.join(DATA, "loan_monthly_performance_train.csv")
TEST   = os.path.join(DATA, "loan_monthly_performance_test.csv")
STATIC = os.path.join(DATA, "loan_static_attributes.csv")
SERV   = os.path.join(DATA, "servicer_updates.csv")
RULES  = "e:/intain/validation_rules.json"
DICT   = "e:/intain/data_dictionary.md"

issues = []   # (severity, file, issue_description, detail)

def add_issue(sev, fname, desc, detail=""):
    issues.append((sev, fname, desc, detail))

# ---------------------------------------------------------------
# 1. LOAD ALL FILES
# ---------------------------------------------------------------
print("=" * 70)
print("LOADING FILES")
print("=" * 70)

train  = pd.read_csv(TRAIN)
test   = pd.read_csv(TEST)
static = pd.read_csv(STATIC)
serv   = pd.read_csv(SERV)
with open(RULES) as f:
    rules = json.load(f)

print(f"  train  : {train.shape[0]:>10,} rows x {train.shape[1]} cols")
print(f"  test   : {test.shape[0]:>10,} rows x {test.shape[1]} cols")
print(f"  static : {static.shape[0]:>10,} rows x {static.shape[1]} cols")
print(f"  servicer: {serv.shape[0]:>10,} rows x {serv.shape[1]} cols")
print()

# ---------------------------------------------------------------
# 2. COLUMN INVENTORY
# ---------------------------------------------------------------
print("=" * 70)
print("COLUMN INVENTORY")
print("=" * 70)
print(f"  train  cols : {list(train.columns)}")
print(f"  test   cols : {list(test.columns)}")
print(f"  static cols : {list(static.columns)}")
print(f"  serv   cols : {list(serv.columns)}")
print()

# Check: test should NOT have target columns
target_cols_in_test = [c for c in ['target_default', 'target_prepay'] if c in test.columns]
if target_cols_in_test:
    add_issue("CRITICAL", "test", f"Target columns present in test: {target_cols_in_test}", "Leakage vector")
else:
    print("  [OK] Test file does not contain target columns.")

# Check: train must have targets
missing_targets = [c for c in ['target_default', 'target_prepay'] if c not in train.columns]
if missing_targets:
    add_issue("CRITICAL", "train", f"Missing target columns: {missing_targets}")
else:
    print("  [OK] Train file contains both target columns.")
print()

# ---------------------------------------------------------------
# 3. SCHEMA CONSISTENCY (train columns ⊇ test columns)
# ---------------------------------------------------------------
print("=" * 70)
print("SCHEMA CONSISTENCY")
print("=" * 70)
test_only = set(test.columns) - set(train.columns)
if test_only:
    add_issue("HIGH", "test", f"Columns in test but not train: {test_only}")
    print(f"  [FAIL] Test has extra columns: {test_only}")
else:
    print("  [OK] All test columns are present in train.")

train_features = set(train.columns) - {'target_default', 'target_prepay'}
missing_in_test = train_features - set(test.columns)
if missing_in_test:
    add_issue("HIGH", "test", f"Feature columns in train but not test: {missing_in_test}")
    print(f"  [FAIL] Test is missing feature columns: {missing_in_test}")
else:
    print("  [OK] All train feature columns are present in test.")
print()

# ---------------------------------------------------------------
# 4. DTYPE CORRECTNESS
# ---------------------------------------------------------------
print("=" * 70)
print("DTYPE CHECKS")
print("=" * 70)
for name, df in [("train", train), ("test", test), ("static", static), ("servicer", serv)]:
    print(f"  --- {name} ---")
    for c in df.columns:
        print(f"    {c:>30s}  {str(df[c].dtype):<12s}  non-null={df[c].notna().sum():>10,}  nulls={df[c].isna().sum():>8,}")
    print()

# Specific type issues
# reporting_period should be string-like MMYYYY but is int
if train['reporting_period'].dtype in [np.int64, np.float64]:
    sample_periods = train['reporting_period'].dropna().unique()[:10]
    # Check format: MMYYYY would be 5-6 digits. If we see values like 12025, 22025, that's MYYYY (no leading zero)
    short_periods = [p for p in sample_periods if p < 100000]
    if short_periods:
        add_issue("MEDIUM", "train", 
                  "reporting_period stored as integer, losing leading zeros (e.g., 12025 instead of '012025')",
                  f"Sample values: {sorted(sample_periods[:10])}")
        print(f"  [WARN] reporting_period as int loses leading zeros. Samples: {sorted(sample_periods[:10])}")

# loan_age should be >= 0
neg_age = train[train['loan_age'] < 0]
if len(neg_age) > 0:
    add_issue("HIGH", "train", 
              f"Negative loan_age found: {len(neg_age)} rows",
              f"loan_ids: {neg_age['loan_id'].unique()[:10].tolist()}, min age: {neg_age['loan_age'].min()}")
    print(f"  [FAIL] {len(neg_age)} rows have negative loan_age. Min = {neg_age['loan_age'].min()}")

neg_age_test = test[test['loan_age'] < 0]
if len(neg_age_test) > 0:
    add_issue("HIGH", "test", f"Negative loan_age found: {len(neg_age_test)} rows")

print()

# ---------------------------------------------------------------
# 5. MISSING VALUES
# ---------------------------------------------------------------
print("=" * 70)
print("MISSING VALUES")
print("=" * 70)

for name, df in [("train", train), ("test", test), ("static", static), ("servicer", serv)]:
    miss = df.isnull().sum()
    miss = miss[miss > 0]
    if len(miss) > 0:
        print(f"  --- {name} ---")
        for c, v in miss.items():
            pct = v / len(df) * 100
            print(f"    {c:>30s}: {v:>10,} missing ({pct:.2f}%)")
            if pct > 50:
                add_issue("HIGH", name, f"Column '{c}' is >50% missing ({pct:.1f}%)", f"{v}/{len(df)} null")
            elif pct > 5:
                add_issue("MEDIUM", name, f"Column '{c}' has significant missingness ({pct:.1f}%)")
    else:
        print(f"  --- {name} --- : no missing values")
    print()

# Specific: zero_balance_code is almost entirely empty.
# Per data dictionary, it should only be populated when UPB = 0.
zbc_train = train['zero_balance_code']
upb_zero = train[train['current_upb'] == 0]
upb_nonzero = train[train['current_upb'] != 0]
zbc_when_upb_zero = upb_zero['zero_balance_code'].notna().sum()
zbc_when_upb_nonzero = upb_nonzero['zero_balance_code'].notna().sum()
print(f"  zero_balance_code when UPB=0: {zbc_when_upb_zero} populated out of {len(upb_zero)}")
print(f"  zero_balance_code when UPB!=0: {zbc_when_upb_nonzero} populated out of {len(upb_nonzero)}")
if len(upb_zero) > 0 and zbc_when_upb_zero == 0:
    add_issue("HIGH", "train", 
              "zero_balance_code is NEVER populated even when current_upb == 0 (violates validation rule 'closed_prepaid_status')",
              f"{len(upb_zero)} rows with UPB=0 have no zero_balance_code")
print()

# ---------------------------------------------------------------
# 6. DUPLICATE ROWS & DUPLICATE KEYS
# ---------------------------------------------------------------
print("=" * 70)
print("DUPLICATE CHECKS")
print("=" * 70)

for name, df, keys in [("train", train, ['loan_id', 'reporting_period']),
                        ("test", test, ['loan_id', 'reporting_period']),
                        ("static", static, ['loan_id'])]:
    total_dups = df.duplicated().sum()
    key_dups   = df.duplicated(subset=keys).sum()
    print(f"  {name}: exact duplicates = {total_dups}, key duplicates ({keys}) = {key_dups}")
    if total_dups > 0:
        add_issue("HIGH", name, f"{total_dups} exact duplicate rows found")
    if key_dups > 0:
        add_issue("CRITICAL", name, f"{key_dups} duplicate keys on {keys} — breaks panel grain",
                  f"Example loan_ids: {df[df.duplicated(subset=keys, keep=False)]['loan_id'].unique()[:5].tolist()}")

# Servicer duplicates
serv_dups = serv.duplicated(subset=['loan_id', 'reporting_period']).sum()
print(f"  servicer: key duplicates (loan_id, reporting_period) = {serv_dups}")
if serv_dups > 0:
    add_issue("MEDIUM", "servicer", f"{serv_dups} duplicate (loan_id, reporting_period) keys in servicer updates")
print()

# ---------------------------------------------------------------
# 7. IMPOSSIBLE NUMERIC VALUES
# ---------------------------------------------------------------
print("=" * 70)
print("IMPOSSIBLE NUMERIC VALUES")
print("=" * 70)

# current_upb should be >= 0
neg_upb = train[train['current_upb'] < 0]
if len(neg_upb) > 0:
    add_issue("CRITICAL", "train", f"{len(neg_upb)} rows with negative current_upb")
    print(f"  [FAIL] {len(neg_upb)} negative current_upb values")
else:
    print("  [OK] No negative current_upb in train")

# interest rate sanity: should be between 0 and 20
ir_bad = train[(train['current_interest_rate'] < 0) | (train['current_interest_rate'] > 20)]
if len(ir_bad) > 0:
    add_issue("HIGH", "train", f"{len(ir_bad)} rows with interest rate outside [0, 20]")
    print(f"  [FAIL] {len(ir_bad)} interest rates outside [0, 20]")
else:
    print("  [OK] Interest rates within reasonable range")

# remaining_months should be >= 0
neg_rem = train[train['remaining_months'] < 0]
if len(neg_rem) > 0:
    add_issue("HIGH", "train", f"{len(neg_rem)} rows with negative remaining_months")
    print(f"  [FAIL] {len(neg_rem)} negative remaining_months values")
else:
    print("  [OK] No negative remaining_months in train")

# delinquency_status should be >= 0
neg_dlq = train[train['delinquency_status'] < 0]
if len(neg_dlq) > 0:
    add_issue("HIGH", "train", f"{len(neg_dlq)} rows with negative delinquency_status")
    print(f"  [FAIL] {len(neg_dlq)} negative delinquency values")
else:
    print("  [OK] No negative delinquency_status")

# Static: LTV should be 1-200, DTI should be 0-65, credit_score 300-850
ltv_bad = static[(static['ltv'] < 1) | (static['ltv'] > 200)]
dti_bad = static[(static['dti'] < 0) | (static['dti'] > 65)]
cs_bad  = static[(static['credit_score'].notna()) & ((static['credit_score'] < 300) | (static['credit_score'] > 850))]
print(f"  static: LTV out of [1,200] = {len(ltv_bad)}, DTI out of [0,65] = {len(dti_bad)}, credit_score out of [300,850] = {len(cs_bad)}")
if len(ltv_bad) > 0:
    add_issue("MEDIUM", "static", f"{len(ltv_bad)} LTV values outside [1, 200]")
if len(dti_bad) > 0:
    add_issue("MEDIUM", "static", f"{len(dti_bad)} DTI values outside [0, 65]")
if len(cs_bad) > 0:
    add_issue("MEDIUM", "static", f"{len(cs_bad)} credit scores outside [300, 850]")
print()

# ---------------------------------------------------------------
# 8. CROSS-COLUMN CONSISTENCY
# ---------------------------------------------------------------
print("=" * 70)
print("CROSS-COLUMN CONSISTENCY")
print("=" * 70)

# loan_age + remaining_months should roughly equal original term
# Original term is not directly in monthly, but we can check:
# loan_age + remaining_months should be constant per loan if term doesn't change
train_check = train.copy()
train_check['total_term'] = train_check['loan_age'] + train_check['remaining_months']
term_var = train_check.groupby('loan_id')['total_term'].agg(['min', 'max'])
term_var['diff'] = term_var['max'] - term_var['min']
inconsistent_term = term_var[term_var['diff'] > 1]
print(f"  Loans where (loan_age + remaining_months) varies by more than 1: {len(inconsistent_term)}")
if len(inconsistent_term) > 0:
    add_issue("HIGH", "train", 
              f"{len(inconsistent_term)} loans have inconsistent (loan_age + remaining_months) — implies term is changing or data is wrong",
              f"Example loan_ids: {inconsistent_term.head(5).index.tolist()}")

# target_default = 1 should imply delinquency_status > 3
default_but_current = train[(train['target_default'] == 1) & (train['delinquency_status'] <= 3)]
print(f"  Rows where target_default=1 but delinquency_status <= 3: {len(default_but_current)}")
if len(default_but_current) > 0:
    add_issue("HIGH", "train",
              f"{len(default_but_current)} rows marked as default (target_default=1) but delinquency_status <= 3",
              "Target definition may be inconsistent")

# target_prepay = 1 should imply current_upb == 0
prepay_with_balance = train[(train['target_prepay'] == 1) & (train['current_upb'] > 0)]
print(f"  Rows where target_prepay=1 but current_upb > 0: {len(prepay_with_balance)}")
if len(prepay_with_balance) > 0:
    add_issue("HIGH", "train",
              f"{len(prepay_with_balance)} rows marked as prepaid (target_prepay=1) but current_upb > 0",
              "Target definition may be inconsistent")

# If both target_default and target_prepay are 1 at the same time — impossible
both_targets = train[(train['target_default'] == 1) & (train['target_prepay'] == 1)]
print(f"  Rows where both target_default=1 AND target_prepay=1: {len(both_targets)}")
if len(both_targets) > 0:
    add_issue("CRITICAL", "train",
              f"{len(both_targets)} rows have BOTH target_default=1 and target_prepay=1 — mutually exclusive labels",
              f"loan_ids: {both_targets['loan_id'].unique()[:10].tolist()}")
print()

# ---------------------------------------------------------------
# 9. VALIDATION RULE CHECKS
# ---------------------------------------------------------------
print("=" * 70)
print("VALIDATION RULE CHECKS")
print("=" * 70)

# Rule 1: balance_consistency — current_upb <= orig_balance * 1.05
merged = train.merge(static[['loan_id', 'orig_upb']], on='loan_id', how='left')
balance_violations = merged[merged['current_upb'] > merged['orig_upb'] * 1.05]
print(f"  Balance consistency violations (UPB > orig_upb*1.05): {len(balance_violations)}")
if len(balance_violations) > 0:
    add_issue("HIGH", "train+static", 
              f"{len(balance_violations)} rows where current_upb exceeds orig_upb by more than 5%",
              f"Max ratio: {(balance_violations['current_upb'] / balance_violations['orig_upb']).max():.3f}")

# Rule 2: delinquency_progression — no jumps > 1
sorted_train = train.sort_values(['loan_id', 'reporting_period'])
sorted_train['prev_dlq'] = sorted_train.groupby('loan_id')['delinquency_status'].shift(1)
dlq_jumps = sorted_train[(sorted_train['prev_dlq'].notna()) & 
                          (sorted_train['delinquency_status'] > sorted_train['prev_dlq'] + 1)]
print(f"  Delinquency progression violations (jump > 1): {len(dlq_jumps)}")
if len(dlq_jumps) > 0:
    add_issue("HIGH", "train", 
              f"{len(dlq_jumps)} rows where delinquency jumped by more than 1 in a single period",
              f"Example: {dlq_jumps[['loan_id','reporting_period','delinquency_status','prev_dlq']].head(5).to_string(index=False)}")

# Rule 3: closed/prepaid — if UPB == 0, zero_balance_code must be populated
upb_zero_rows = train[train['current_upb'] == 0]
zbc_missing_when_zero = upb_zero_rows[upb_zero_rows['zero_balance_code'].isna()]
print(f"  Closed/prepaid violations (UPB=0, zero_balance_code missing): {len(zbc_missing_when_zero)} out of {len(upb_zero_rows)} zero-UPB rows")
if len(zbc_missing_when_zero) > 0:
    add_issue("HIGH", "train",
              f"{len(zbc_missing_when_zero)} rows with current_upb=0 but zero_balance_code is missing",
              "Violates validation rule 'closed_prepaid_status'")
print()

# ---------------------------------------------------------------
# 10. LEAKAGE RISK CHECK
# ---------------------------------------------------------------
print("=" * 70)
print("LEAKAGE RISK CHECK")
print("=" * 70)

# delinquency_status is the direct signal for target_default
# If target_default = (delinquency_status > 3), that's a tautological leak
corr_default_dlq = train['target_default'].corr(train['delinquency_status'])
print(f"  Correlation(target_default, delinquency_status) = {corr_default_dlq:.4f}")

# Check if delinquency_status perfectly determines target_default
dlq_gt3 = (train['delinquency_status'] > 3).astype(int)
perfect_match = (dlq_gt3 == train['target_default']).all()
print(f"  target_default == (delinquency_status > 3)?  {perfect_match}")
if perfect_match:
    add_issue("CRITICAL", "train",
              "target_default is EXACTLY equal to (delinquency_status > 3) — TAUTOLOGICAL LEAKAGE",
              "delinquency_status encodes the target. Must be lagged or excluded.")

# Check if target_prepay is exactly (current_upb == 0)
upb_zero_flag = (train['current_upb'] == 0).astype(int)
prepay_match = (upb_zero_flag == train['target_prepay']).all()
print(f"  target_prepay == (current_upb == 0)?  {prepay_match}")
if prepay_match:
    add_issue("CRITICAL", "train",
              "target_prepay is EXACTLY equal to (current_upb == 0) — TAUTOLOGICAL LEAKAGE",
              "current_upb encodes the prepay target. Must be lagged or excluded.")

# zero_balance_code leakage: populated only when loan terminates
zbc_notnull = train['zero_balance_code'].notna()
if zbc_notnull.sum() > 0:
    zbc_and_target = train[zbc_notnull][['target_default', 'target_prepay']].sum()
    print(f"  When zero_balance_code is populated: {zbc_and_target.to_dict()}")
    add_issue("HIGH", "train",
              "zero_balance_code is a post-event indicator — must be excluded from features",
              "It is only set when a loan terminates, directly encoding the event")

print()

# ---------------------------------------------------------------
# 11. JOIN CONSISTENCY
# ---------------------------------------------------------------
print("=" * 70)
print("JOIN CONSISTENCY")
print("=" * 70)

train_ids = set(train['loan_id'].unique())
test_ids  = set(test['loan_id'].unique())
static_ids = set(static['loan_id'].unique())
serv_ids   = set(serv['loan_id'].unique())

train_not_in_static = train_ids - static_ids
test_not_in_static  = test_ids - static_ids
serv_not_in_train   = serv_ids - train_ids
train_test_overlap  = train_ids & test_ids

print(f"  Unique loan_ids: train={len(train_ids)}, test={len(test_ids)}, static={len(static_ids)}, servicer={len(serv_ids)}")
print(f"  Train loans NOT in static: {len(train_not_in_static)}")
print(f"  Test loans NOT in static:  {len(test_not_in_static)}")
print(f"  Servicer loans NOT in train: {len(serv_not_in_train)}")
print(f"  Train-Test loan_id overlap: {len(train_test_overlap)}")

if len(train_not_in_static) > 0:
    add_issue("HIGH", "train+static", f"{len(train_not_in_static)} train loans have no static attributes")
if len(test_not_in_static) > 0:
    add_issue("HIGH", "test+static", f"{len(test_not_in_static)} test loans have no static attributes")
if len(train_test_overlap) == 0:
    add_issue("MEDIUM", "train+test", "No loan_id overlap between train and test — chronological split may lose panel continuity")
elif len(train_test_overlap) > 0:
    print(f"  [OK] {len(train_test_overlap)} loans appear in both train and test (expected for panel data with temporal split)")

# Servicer join: check conflicts
serv_merged = train.merge(serv, on=['loan_id', 'reporting_period'], suffixes=('', '_serv'), how='inner')
print(f"\n  Servicer-Train inner join rows: {len(serv_merged)}")
if len(serv_merged) > 0:
    # Check delinquency conflicts
    serv_merged['dlq_conflict'] = (serv_merged['delinquency_status'] != serv_merged['delinquency_status_serv'])
    dlq_conflicts = serv_merged['dlq_conflict'].sum()
    
    # Check UPB conflicts (within 1% tolerance)
    serv_merged['upb_pct_diff'] = abs(serv_merged['current_upb'] - serv_merged['current_upb_serv']) / (serv_merged['current_upb'] + 1e-9)
    upb_conflicts = (serv_merged['upb_pct_diff'] > 0.01).sum()
    
    print(f"  Delinquency conflicts with servicer: {dlq_conflicts}")
    print(f"  UPB conflicts with servicer (>1% diff): {upb_conflicts}")
    
    if dlq_conflicts > 0:
        add_issue("HIGH", "servicer", 
                  f"{dlq_conflicts} delinquency conflicts between primary and servicer data",
                  f"Conflict rate: {dlq_conflicts/len(serv_merged):.1%}")
    if upb_conflicts > 0:
        add_issue("MEDIUM", "servicer",
                  f"{upb_conflicts} UPB conflicts between primary and servicer data (>1% difference)")

# Servicer: all delinquency_status values are NaN — check
serv_dlq_null = serv['delinquency_status'].isna().sum()
print(f"\n  Servicer delinquency_status null: {serv_dlq_null}/{len(serv)} ({serv_dlq_null/len(serv)*100:.1f}%)")
if serv_dlq_null == len(serv):
    add_issue("CRITICAL", "servicer",
              "servicer_updates.csv has delinquency_status 100% NULL — file is useless for conflict detection",
              "The entire column is empty. No conflict detection is possible.")

# Servicer: stale update_date
serv_dates = serv['update_date'].unique()
print(f"  Servicer update_date unique values: {serv_dates}")
if len(serv_dates) == 1:
    add_issue("HIGH", "servicer",
              f"All servicer updates have the SAME update_date ({serv_dates[0]}) — synthetic/mock data",
              "Real servicer updates would have varying dates")
print()

# ---------------------------------------------------------------
# 12. TRAIN-TEST DRIFT
# ---------------------------------------------------------------
print("=" * 70)
print("TRAIN-TEST DRIFT")
print("=" * 70)

shared_num_cols = [c for c in train.select_dtypes(include=[np.number]).columns 
                   if c in test.columns and c not in ['target_default', 'target_prepay']]

for col in shared_num_cols:
    t_mean = train[col].mean()
    e_mean = test[col].mean()
    t_std  = train[col].std()
    e_std  = test[col].std()
    if t_mean != 0:
        pct_diff = abs(t_mean - e_mean) / abs(t_mean) * 100
    else:
        pct_diff = 0
    flag = " *** DRIFT" if pct_diff > 10 else ""
    print(f"  {col:>25s}  train_mean={t_mean:>12.2f}  test_mean={e_mean:>12.2f}  drift={pct_diff:>6.1f}%{flag}")
    if pct_diff > 20:
        add_issue("MEDIUM", "train vs test", f"Significant drift in '{col}': {pct_diff:.1f}% mean difference")
print()

# Temporal check: test should be strictly after train
train_max_period = train['reporting_period'].max()
test_min_period  = test['reporting_period'].min()
test_max_period  = test['reporting_period'].max()
print(f"  Train max period: {train_max_period}")
print(f"  Test period range: {test_min_period} to {test_max_period}")
if test_min_period <= train_max_period:
    add_issue("CRITICAL", "train+test", 
              f"Test period ({test_min_period}) overlaps with train max ({train_max_period}) — temporal leakage risk")
else:
    print("  [OK] Test is strictly after train.")
print()

# ---------------------------------------------------------------
# 13. OUTLIERS AND EXTREME VALUES
# ---------------------------------------------------------------
print("=" * 70)
print("OUTLIERS AND EXTREME VALUES")
print("=" * 70)

for col in ['current_upb', 'current_interest_rate', 'loan_age', 'remaining_months', 'delinquency_status']:
    data = train[col].dropna()
    if len(data) == 0:
        continue
    q1, q3 = data.quantile(0.25), data.quantile(0.75)
    iqr = q3 - q1
    low  = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    n_outlier = ((data < low) | (data > high)).sum()
    pct = n_outlier / len(data) * 100
    print(f"  {col:>25s}: Q1={q1:>10.2f} Q3={q3:>10.2f} IQR={iqr:>10.2f}  outliers={n_outlier:>8,} ({pct:.2f}%)")
    print(f"  {'':>25s}  min={data.min():>12.2f}  max={data.max():>12.2f}")
    if pct > 5:
        add_issue("MEDIUM", "train", f"'{col}' has {pct:.1f}% outliers (outside 1.5×IQR)")

# Static outliers
for col in ['orig_upb', 'credit_score', 'ltv', 'dti']:
    data = static[col].dropna()
    if len(data) == 0:
        continue
    q1, q3 = data.quantile(0.25), data.quantile(0.75)
    iqr = q3 - q1
    low  = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    n_outlier = ((data < low) | (data > high)).sum()
    pct = n_outlier / len(data) * 100
    print(f"  static.{col:>18s}: outliers={n_outlier:>6,} ({pct:.2f}%), min={data.min():.1f}, max={data.max():.1f}")
print()

# ---------------------------------------------------------------
# 14. TARGET DISTRIBUTION
# ---------------------------------------------------------------
print("=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)

for t in ['target_default', 'target_prepay']:
    if t in train.columns:
        vc = train[t].value_counts()
        print(f"  {t}:")
        for val, cnt in vc.items():
            print(f"    {val}: {cnt:>10,} ({cnt/len(train)*100:.3f}%)")
        if vc.get(1, 0) == 0:
            add_issue("CRITICAL", "train", f"target '{t}' has ZERO positive cases — cannot train a model")
        elif vc.get(1, 0) / len(train) < 0.001:
            add_issue("HIGH", "train", f"target '{t}' is extremely rare ({vc.get(1,0)/len(train)*100:.3f}%) — severe class imbalance")
print()

# ---------------------------------------------------------------
# 15. FINAL ISSUE SUMMARY
# ---------------------------------------------------------------
print("=" * 70)
print("FINAL ISSUE SUMMARY")
print("=" * 70)

# Sort by severity
sev_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
issues.sort(key=lambda x: sev_order.get(x[0], 99))

critical_count = sum(1 for i in issues if i[0] == 'CRITICAL')
high_count     = sum(1 for i in issues if i[0] == 'HIGH')
medium_count   = sum(1 for i in issues if i[0] == 'MEDIUM')

print(f"\nTotal issues: {len(issues)}")
print(f"  CRITICAL: {critical_count}")
print(f"  HIGH    : {high_count}")
print(f"  MEDIUM  : {medium_count}")
print()

for idx, (sev, fname, desc, detail) in enumerate(issues, 1):
    print(f"  [{sev}] #{idx} ({fname}): {desc}")
    if detail:
        print(f"         Detail: {detail}")

# ---------------------------------------------------------------
# 16. FILE-BY-FILE VERDICT
# ---------------------------------------------------------------
print()
print("=" * 70)
print("FILE-BY-FILE VERDICT")
print("=" * 70)

file_verdicts = {}
for fname_key in ['train', 'test', 'static', 'servicer', 'train+static', 'test+static', 'train+test', 'train vs test']:
    file_issues = [i for i in issues if i[1] == fname_key]
    has_critical = any(i[0] == 'CRITICAL' for i in file_issues)
    has_high = any(i[0] == 'HIGH' for i in file_issues)
    if has_critical:
        verdict = "FAIL"
    elif has_high:
        verdict = "CONDITIONAL FAIL"
    else:
        verdict = "PASS"
    file_verdicts[fname_key] = verdict

for fname, verdict in file_verdicts.items():
    n = sum(1 for i in issues if i[1] == fname)
    if n > 0:
        print(f"  {fname:>20s}: {verdict} ({n} issues)")

# ---------------------------------------------------------------
# 17. OVERALL VERDICT
# ---------------------------------------------------------------
print()
print("=" * 70)
print("OVERALL MODELING READINESS VERDICT")
print("=" * 70)

if critical_count > 0:
    print(f"\n  *** VERDICT: NOT READY FOR MODELING ***")
    print(f"  Reason: {critical_count} CRITICAL issues remain.")
    print(f"  The dataset contains tautological leakage, broken servicer data,")
    print(f"  and structural issues that would produce invalid model results.")
    print(f"  These must be resolved before proceeding to Task 2.")
else:
    print(f"\n  *** VERDICT: CONDITIONALLY READY ***")
    print(f"  {high_count} HIGH issues should be addressed first.")

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)
