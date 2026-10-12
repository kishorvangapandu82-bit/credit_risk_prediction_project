"""
=============================================================================
V2 PIPELINE -- STEP 1: DECISION THRESHOLDS & EXTRA METRICS
=============================================================================
Computes:
1. Training-set Out-Of-Fold (5-fold CV) threshold selection:
   - Optimal threshold maximizing F1 on default class (class 1)
   - Youden's J statistic threshold (max TPR - FPR)
2. Evaluation on untouched holdout test set:
   - Performance at threshold = 0.50
   - Performance at chosen training-set OOF F1 threshold
   - Performance at Youden's J threshold
3. Extra metrics on holdout test set:
   - ROC-AUC
   - Gini Coefficient (2 * AUC - 1)
   - Average Precision / PR-AUC
   - Kolmogorov-Smirnov (KS) statistic
   - Brier Score
4. Visualizations:
   - Calibration curves (Top 3: LightGBM, XGBoost, Random Forest)
   - Precision-Recall curves (All 7 models)
   Saved in PNG (300 dpi) and PDF.
=============================================================================
"""

import os, sys, joblib, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, average_precision_score, recall_score, precision_score,
    f1_score, accuracy_score, brier_score_loss, roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve

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
FIG_DIR   = "outputs_v2/figures"
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# 1. Load, clean, split
print("[Step 1] Loading and splitting dataset...")
df_raw = load_data()
df_clean, _ = clean_data(df_raw, verbose=False)
X_train, X_test, y_train, y_test = split_data(df_clean)

# Baseline models configurations matching Table II in paper
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
    else:
        raise ValueError(f"Unknown model: {name}")

MODEL_SPECS = [
    ("LightGBM",            "lightgbm.joblib",            "#17becf"),
    ("XGBoost",             "xgboost.joblib",             "#2ca02c"),
    ("Random Forest",       "random_forest.joblib",       "#8c564b"),
    ("Decision Tree",       "decision_tree.joblib",       "#ff7f0e"),
    ("MLP",                 "mlp.joblib",                 "#9467bd"),
    ("Logistic Regression", "logistic_regression.joblib", "#1f77b4"),
    ("SVM",                 "svm.joblib",                 "#d62728"),
]

# Function to calculate KS statistic
def compute_ks(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(np.abs(tpr - fpr)))

# 2. Run OOF predictions on training set for threshold tuning
print("\n[Step 2] Generating 5-fold CV Out-of-Fold (OOF) predictions on training data...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
threshold_search = np.linspace(0.01, 0.99, 99)

oof_probs_dict = {}
chosen_thresholds = []

for name, fname, color in MODEL_SPECS:
    print(f"  -> Generating OOF probabilities for {name}...")
    pipe = get_fresh_pipeline(name)
    # MLP requires sample weights for imbalance if specified, but baseline pipeline uses unweighted MLP fit
    oof_prob = cross_val_predict(pipe, X_train, y_train, cv=skf, method="predict_proba", n_jobs=-1)[:, 1]
    oof_probs_dict[name] = oof_prob

    # Optimize threshold on OOF predictions
    best_f1 = -1.0
    best_t_f1 = 0.50
    for t in threshold_search:
        pred_c1 = (oof_prob >= t).astype(int)
        f1 = f1_score(y_train, pred_c1, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t_f1 = t

    # Youden's J statistic threshold on OOF: argmax (TPR - FPR)
    fpr_oof, tpr_oof, thresh_roc = roc_curve(y_train, oof_prob)
    j_scores = tpr_oof - fpr_oof
    best_j_idx = np.argmax(j_scores)
    best_t_youden = thresh_roc[best_j_idx]
    if np.isinf(best_t_youden):
        best_t_youden = 0.50

    chosen_thresholds.append({
        "Model": name,
        "OOF_Best_F1": round(best_f1, 4),
        "Optimal_F1_Threshold": round(best_t_f1, 4),
        "Youden_J_Threshold": round(best_t_youden, 4),
    })

thresh_df = pd.DataFrame(chosen_thresholds)
thresh_df.to_csv(os.path.join(TABLE_DIR, "thresholds.csv"), index=False)
print("\n=== Chosen Training-Set OOF Thresholds ===")
print(thresh_df.to_string())

# 3. Evaluate holdout test set under both 0.50 and chosen threshold
print("\n[Step 3] Evaluating holdout test set at 0.50 and OOF chosen threshold...")
table_iv_v2_rows = []
extra_metrics_rows = []
test_probs_dict = {}

for name, fname, color in MODEL_SPECS:
    model_path = os.path.join("models", fname)
    m = joblib.load(model_path)
    y_prob = m.predict_proba(X_test)[:, 1]
    test_probs_dict[name] = y_prob

    opt_thresh = thresh_df.loc[thresh_df["Model"] == name, "Optimal_F1_Threshold"].values[0]
    youden_thresh = thresh_df.loc[thresh_df["Model"] == name, "Youden_J_Threshold"].values[0]

    # Metrics at 0.50
    pred_050 = (y_prob >= 0.50).astype(int)
    rec_050  = recall_score(y_test, pred_050, zero_division=0)
    prec_050 = precision_score(y_test, pred_050, zero_division=0)
    f1_050   = f1_score(y_test, pred_050, zero_division=0)
    acc_050  = accuracy_score(y_test, pred_050)

    # Metrics at Optimal OOF F1 threshold
    pred_opt = (y_prob >= opt_thresh).astype(int)
    rec_opt  = recall_score(y_test, pred_opt, zero_division=0)
    prec_opt = precision_score(y_test, pred_opt, zero_division=0)
    f1_opt   = f1_score(y_test, pred_opt, zero_division=0)
    acc_opt  = accuracy_score(y_test, pred_opt)

    # Metrics at Youden threshold
    pred_ydn = (y_prob >= youden_thresh).astype(int)
    rec_ydn  = recall_score(y_test, pred_ydn, zero_division=0)
    prec_ydn = precision_score(y_test, pred_ydn, zero_division=0)
    f1_ydn   = f1_score(y_test, pred_ydn, zero_division=0)
    acc_ydn  = accuracy_score(y_test, pred_ydn)

    auc = roc_auc_score(y_test, y_prob)

    table_iv_v2_rows.append({
        "Model": name,
        "ROC_AUC": round(auc, 4),
        "Cutoff_0.50_Recall": round(rec_050, 4),
        "Cutoff_0.50_Precision": round(prec_050, 4),
        "Cutoff_0.50_F1": round(f1_050, 4),
        "Cutoff_0.50_Accuracy": round(acc_050, 4),
        "Optimal_Threshold": round(opt_thresh, 4),
        "OptThresh_Recall": round(rec_opt, 4),
        "OptThresh_Precision": round(prec_opt, 4),
        "OptThresh_F1": round(f1_opt, 4),
        "OptThresh_Accuracy": round(acc_opt, 4),
        "Youden_Threshold": round(youden_thresh, 4),
        "Youden_Recall": round(rec_ydn, 4),
        "Youden_Precision": round(prec_ydn, 4),
        "Youden_F1": round(f1_ydn, 4),
        "Youden_Accuracy": round(acc_ydn, 4),
    })

    # Extra Metrics
    gini = 2.0 * auc - 1.0
    ap = average_precision_score(y_test, y_prob)
    ks = compute_ks(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)

    extra_metrics_rows.append({
        "Model": name,
        "ROC_AUC": round(auc, 4),
        "Gini_Coefficient": round(gini, 4),
        "PR_AUC_Average_Precision": round(ap, 4),
        "KS_Statistic": round(ks, 4),
        "Brier_Score": round(brier, 4),
    })

table_iv_v2_df = pd.DataFrame(table_iv_v2_rows)
table_iv_v2_df.to_csv(os.path.join(TABLE_DIR, "table_iv_v2.csv"), index=False)
print("\n=== Table IV v2 (Holdout Performance at 0.50 and Optimized Thresholds) ===")
print(table_iv_v2_df.to_string())

extra_metrics_df = pd.DataFrame(extra_metrics_rows)
extra_metrics_df.to_csv(os.path.join(TABLE_DIR, "extra_metrics.csv"), index=False)
print("\n=== Extra Metrics (ROC-AUC, Gini, PR-AUC, KS, Brier Score) ===")
print(extra_metrics_df.to_string())

# 4. Generate Figures: PR Curves & Calibration Curves
print("\n[Step 4] Generating publication-grade figures (PNG 300 dpi + PDF)...")
plt.rcParams.update({"font.size": 9, "font.family": "serif", "figure.dpi": 300})

# A. PR Curves (All 7 models)
fig, ax = plt.subplots(figsize=(6.5, 4.5))
for name, fname, color in MODEL_SPECS:
    y_prob = test_probs_dict[name]
    ap = average_precision_score(y_test, y_prob)
    prec_vals, rec_vals, _ = precision_recall_curve(y_test, y_prob)
    ax.plot(rec_vals, prec_vals, label=f"{name} (AP = {ap:.4f})", color=color, linewidth=1.5)

# Baseline default rate reference line
base_rate = y_test.mean()
ax.axhline(base_rate, color="gray", linestyle="--", linewidth=1.0, label=f"Random Baseline ({base_rate*100:.2f}%)")

ax.set_xlabel("Recall (Class 1 - Serious Delinquency)", fontsize=10)
ax.set_ylabel("Precision (Class 1)", fontsize=10)
ax.set_title("Precision-Recall Curves Across All Seven Benchmark Architectures", fontsize=11, fontweight="bold")
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(loc="upper right", frameon=True, fontsize=8)
plt.tight_layout()

pr_png = os.path.join(FIG_DIR, "precision_recall_curves_v2.png")
pr_pdf = os.path.join(FIG_DIR, "precision_recall_curves_v2.pdf")
plt.savefig(pr_png, dpi=300)
plt.savefig(pr_pdf)
plt.close()
print(f"  Saved PR Curves: {pr_png} and {pr_pdf}")

# B. Calibration Curve (Top 3 models)
fig, ax = plt.subplots(figsize=(5.5, 4.5))
top3 = [("LightGBM", "#17becf"), ("XGBoost", "#2ca02c"), ("Random Forest", "#8c564b")]
ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", linewidth=1.0)

for name, color in top3:
    y_prob = test_probs_dict[name]
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy="quantile")
    brier = brier_score_loss(y_test, y_prob)
    ax.plot(prob_pred, prob_true, marker="s", markersize=4, label=f"{name} (Brier = {brier:.4f})", color=color, linewidth=1.5)

ax.set_xlabel("Mean Predicted Probability", fontsize=10)
ax.set_ylabel("Empirical Fraction of Positives", fontsize=10)
ax.set_title("Reliability Calibration Curve (Top 3 Ensembles)", fontsize=11, fontweight="bold")
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(loc="upper left", frameon=True, fontsize=8.5)
plt.tight_layout()

cal_png = os.path.join(FIG_DIR, "calibration_curves_top3.png")
cal_pdf = os.path.join(FIG_DIR, "calibration_curves_top3.pdf")
plt.savefig(cal_png, dpi=300)
plt.savefig(cal_pdf)
plt.close()
print(f"  Saved Calibration Curves: {cal_png} and {cal_pdf}")

print("\nStep 1 Complete!")
