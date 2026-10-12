"""
=============================================================================
V2 PIPELINE -- STEP 2: STATISTICAL RIGOR & HYPOTHESIS TESTING
=============================================================================
Computes:
1. 1000-sample paired bootstrap 95% Confidence Intervals for holdout ROC-AUC.
2. Pairwise statistical significance tests:
   - LightGBM vs XGBoost
   - LightGBM vs Random Forest
   - XGBoost vs Random Forest
   (Paired bootstrap test + DeLong analytical variance test)
3. Repeated Stratified 5-Fold Cross-Validation x 5 repeats (25 folds)
   on the training partition for all 7 benchmark architectures.
=============================================================================
"""

import os, sys, joblib, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

# Add src to path
sys.path.insert(0, os.path.abspath("src"))
from config import RANDOM_STATE
from data_preprocessing import load_data, clean_data, split_data
from models import (
    build_logistic_pipeline, build_decision_tree_pipeline, build_svm_pipeline,
    build_mlp_pipeline, build_random_forest_pipeline, build_xgboost_pipeline,
    build_lightgbm_pipeline
)

TABLE_DIR = "outputs_v2/tables"
os.makedirs(TABLE_DIR, exist_ok=True)

# 1. Load data
print("[Step 2.1] Loading and preparing dataset...")
df_raw = load_data()
df_clean, _ = clean_data(df_raw, verbose=False)
X_train, X_test, y_train, y_test = split_data(df_clean)

y_test_arr = y_test.values

# Models to evaluate
MODEL_FILES = [
    ("LightGBM",            "lightgbm.joblib"),
    ("XGBoost",             "xgboost.joblib"),
    ("Random Forest",       "random_forest.joblib"),
    ("Decision Tree",       "decision_tree.joblib"),
    ("MLP",                 "mlp.joblib"),
    ("Logistic Regression", "logistic_regression.joblib"),
    ("SVM",                 "svm.joblib"),
]

# Load predictions
print("\n[Step 2.2] Loading models and predicting probabilities on holdout test set...")
probs = {}
for name, fname in MODEL_FILES:
    m = joblib.load(os.path.join("models", fname))
    probs[name] = m.predict_proba(X_test)[:, 1]

# 2. 1000-sample bootstrap 95% Confidence Intervals
print("\n[Step 2.3] Running 1,000-sample paired bootstrap for holdout ROC-AUC...")
N_BOOTSTRAP = 1000
n_test = len(y_test_arr)

np.random.seed(RANDOM_STATE)
boot_indices = [np.random.choice(n_test, size=n_test, replace=True) for _ in range(N_BOOTSTRAP)]

boot_aucs = {name: np.zeros(N_BOOTSTRAP) for name, _ in MODEL_FILES}

for b_idx, idx in enumerate(boot_indices):
    y_b = y_test_arr[idx]
    # Check both classes are present
    if len(np.unique(y_b)) < 2:
        continue
    for name, _ in MODEL_FILES:
        boot_aucs[name][b_idx] = roc_auc_score(y_b, probs[name][idx])

ci_rows = []
for name, _ in MODEL_FILES:
    arr = boot_aucs[name]
    mean_auc = np.mean(arr)
    se_auc   = np.std(arr, ddof=1)
    ci_lower = np.percentile(arr, 2.5)
    ci_upper = np.percentile(arr, 97.5)
    base_auc = roc_auc_score(y_test_arr, probs[name])

    ci_rows.append({
        "Model": name,
        "Holdout_AUC": round(base_auc, 4),
        "Bootstrap_Mean": round(mean_auc, 4),
        "Bootstrap_SE": round(se_auc, 4),
        "CI_95_Lower": round(ci_lower, 4),
        "CI_95_Upper": round(ci_upper, 4),
        "CI_95_Formatted": f"[{ci_lower:.4f}, {ci_upper:.4f}]"
    })

ci_df = pd.DataFrame(ci_rows)
ci_df.to_csv(os.path.join(TABLE_DIR, "bootstrap_ci.csv"), index=False)
print("\n=== Bootstrap 95% Confidence Intervals ===")
print(ci_df.to_string())

# 3. Pairwise Hypothesis Testing (DeLong and Paired Bootstrap)
print("\n[Step 2.4] Running pairwise hypothesis tests (LightGBM vs XGBoost, LightGBM vs RF, XGBoost vs RF)...")

# Fast DeLong AUC covariance implementation
def delong_test(y_true, p_A, p_B):
    """
    Computes DeLong test z-statistic and p-value for two correlated ROC curves.
    """
    pos_mask = (y_true == 1)
    neg_mask = (y_true == 0)
    m = np.sum(pos_mask)
    n = np.sum(neg_mask)
    
    # Placements for Model A
    x_A = p_A[pos_mask]
    y_A = p_A[neg_mask]
    # Placements for Model B
    x_B = p_B[pos_mask]
    y_B = p_B[neg_mask]

    # Structural components
    v10_A = np.mean(x_A[:, None] > y_A[None, :] + 0.5 * (x_A[:, None] == y_A[None, :]), axis=1)
    v01_A = np.mean(x_A[:, None] < y_A[None, :] + 0.5 * (x_A[:, None] == y_A[None, :]), axis=0)

    v10_B = np.mean(x_B[:, None] > y_B[None, :] + 0.5 * (x_B[:, None] == y_B[None, :]), axis=1)
    v01_B = np.mean(x_B[:, None] < y_B[None, :] + 0.5 * (x_B[:, None] == y_B[None, :]), axis=0)

    auc_A = np.mean(v10_A)
    auc_B = np.mean(v10_B)

    # Covariance components
    s10 = np.cov(v10_A, v10_B)
    s01 = np.cov(v01_A, v01_B)

    var_diff = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / m + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n
    se_diff = np.sqrt(max(var_diff, 1e-12))
    z = (auc_A - auc_B) / se_diff
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))
    return auc_A - auc_B, se_diff, z, p_val

pairs = [
    ("LightGBM", "XGBoost"),
    ("LightGBM", "Random Forest"),
    ("XGBoost",  "Random Forest"),
]

sig_rows = []
for m_A, m_B in pairs:
    diff_auc, se_diff, z_val, delong_p = delong_test(y_test_arr, probs[m_A], probs[m_B])
    
    # Paired bootstrap p-value
    delta_boot = boot_aucs[m_A] - boot_aucs[m_B]
    boot_ci_low = np.percentile(delta_boot, 2.5)
    boot_ci_high = np.percentile(delta_boot, 97.5)
    # two-tailed p-value against null delta = 0
    boot_p = 2.0 * min(np.mean(delta_boot <= 0), np.mean(delta_boot >= 0))
    boot_p = min(max(boot_p, 1.0 / N_BOOTSTRAP), 1.0)

    sig_rows.append({
        "Model_A": m_A,
        "Model_B": m_B,
        "AUC_Diff (A - B)": round(diff_auc, 5),
        "DeLong_SE": round(se_diff, 5),
        "DeLong_Z": round(z_val, 4),
        "DeLong_p_value": round(delong_p, 5),
        "Bootstrap_Diff_95_CI": f"[{boot_ci_low:.5f}, {boot_ci_high:.5f}]",
        "Bootstrap_p_value": round(boot_p, 5),
        "Significant_at_0.05": "Yes" if delong_p < 0.05 else "No"
    })

sig_df = pd.DataFrame(sig_rows)
sig_df.to_csv(os.path.join(TABLE_DIR, "significance_tests.csv"), index=False)
print("\n=== Pairwise Statistical Significance Tests ===")
print(sig_df.to_string())

# 4. Repeated Stratified 5-Fold Cross-Validation x 5 Repeats (25 folds)
print("\n[Step 2.5] Running Repeated Stratified 5-Fold CV x 5 repeats (25 folds total) on training data...")
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)

def get_fresh_pipeline(name):
    if name == "LightGBM":
        return build_lightgbm_pipeline(learning_rate=0.05, num_leaves=31, n_estimators=200, scale_pos_weight=13.96)
    elif name == "XGBoost":
        return build_xgboost_pipeline(learning_rate=0.05, max_depth=5, n_estimators=200, scale_pos_weight=13.96)
    elif name == "Random Forest":
        return build_random_forest_pipeline(n_estimators=200, max_depth=12, min_samples_split=50)
    elif name == "Decision Tree":
        return build_decision_tree_pipeline(max_depth=6, min_samples_split=20, criterion="gini")
    elif name == "MLP":
        pipe, _ = build_mlp_pipeline()
        return pipe
    elif name == "Logistic Regression":
        return build_logistic_pipeline(C=0.001)
    elif name == "SVM":
        return build_svm_pipeline(C=0.01)

rcv_rows = []
for name, _ in MODEL_FILES:
    print(f"  -> Repeated CV for {name} (25 folds)...")
    pipe = get_fresh_pipeline(name)
    scores = cross_val_score(pipe, X_train, y_train, cv=rskf, scoring="roc_auc", n_jobs=-1)
    rcv_rows.append({
        "Model": name,
        "CV_Folds": len(scores),
        "Mean_ROC_AUC": round(float(np.mean(scores)), 4),
        "Std_ROC_AUC": round(float(np.std(scores, ddof=1)), 4),
        "Min_ROC_AUC": round(float(np.min(scores)), 4),
        "Max_ROC_AUC": round(float(np.max(scores)), 4),
    })

rcv_df = pd.DataFrame(rcv_rows)
rcv_df.to_csv(os.path.join(TABLE_DIR, "repeated_cv_results.csv"), index=False)
print("\n=== Repeated 5-Fold x 5-Repeat Cross-Validation Results ===")
print(rcv_df.to_string())

print("\nStep 2 Complete!")
