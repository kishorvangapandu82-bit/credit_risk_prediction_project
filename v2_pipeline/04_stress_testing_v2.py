"""
=============================================================================
V2 PIPELINE -- STEP 4: COMPREHENSIVE MACROECONOMIC STRESS TESTING
=============================================================================
Scenarios evaluated on untouched holdout test set (baseline models, no retraining):
  (a) Income x0.80 only, with DebtRatio recomputed as DebtRatio / 0.80
  (b) DebtRatio x1.25 only (nominal debt surge, constant income)
  (c) Both combined (Income x0.80 AND DebtRatio x1.25 -- original double-counted version)
  (d) Income-shock severity sweep: -10%, -20%, -30%, -40% (with recomputed DebtRatio)
  (e) Targeted shock: Income -20% applied only to the 30% of borrowers with lowest income
Explicit handling of missing-income rows documented.
Reports:
  - Stressed ROC-AUC
  - Retained AUC (unrounded percentage)
  - Relative change in MEAN PREDICTED DEFAULT PROBABILITY (%)
  - Relative change in SHARE PREDICTED AS DEFAULT at chosen optimal threshold (%)
Generates Figure 6 (PNG 300 dpi + PDF).
=============================================================================
"""

import os, sys, joblib, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score

# Add src to path
sys.path.insert(0, os.path.abspath("src"))
from data_preprocessing import load_data, clean_data, split_data

TABLE_DIR = "outputs_v2/tables"
FIG_DIR   = "outputs_v2/figures"
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# 1. Load data
print("[Stress Test] Loading data and holdout partition...")
df_raw = load_data()
df_clean, _ = clean_data(df_raw, verbose=False)
_, X_test_base, _, y_test = split_data(df_clean)

# Load chosen thresholds from Step 1
thresh_file = os.path.join(TABLE_DIR, "thresholds.csv")
if os.path.exists(thresh_file):
    thresh_df = pd.read_csv(thresh_file)
    opt_thresholds = dict(zip(thresh_df["Model"], thresh_df["Optimal_F1_Threshold"]))
else:
    opt_thresholds = {m: 0.50 for m in ["LightGBM", "XGBoost", "Random Forest", "Decision Tree", "MLP", "Logistic Regression", "SVM"]}

MODEL_SPECS = [
    ("LightGBM",            "lightgbm.joblib",            "#17becf"),
    ("XGBoost",             "xgboost.joblib",             "#2ca02c"),
    ("Random Forest",       "random_forest.joblib",       "#8c564b"),
    ("Decision Tree",       "decision_tree.joblib",       "#ff7f0e"),
    ("MLP",                 "mlp.joblib",                 "#9467bd"),
    ("Logistic Regression", "logistic_regression.joblib", "#1f77b4"),
    ("SVM",                 "svm.joblib",                 "#d62728"),
]

# Baseline predictions
baseline_metrics = {}
for name, fname, color in MODEL_SPECS:
    m = joblib.load(os.path.join("models", fname))
    p_base = m.predict_proba(X_test_base)[:, 1]
    t = opt_thresholds.get(name, 0.50)
    pred_base = (p_base >= t).astype(int)

    baseline_metrics[name] = {
        "model": m,
        "auc_base": roc_auc_score(y_test, p_base),
        "mean_prob_base": float(np.mean(p_base)),
        "share_def_base": float(np.mean(pred_base)),
        "thresh": t,
        "color": color
    }

# ---------------------------------------------------------------------------
# Scenario Functions
# ---------------------------------------------------------------------------
# Scenario (a): Income x0.80, DebtRatio recomputed as DebtRatio / 0.80
def get_scenario_a(X):
    X_s = X.copy()
    X_s["MonthlyIncome"] = X_s["MonthlyIncome"] * 0.80
    X_s["DebtRatio"]     = X_s["DebtRatio"] / 0.80
    return X_s

# Scenario (a_quirk): Income x0.80, DebtRatio recomputed only where income is present
def get_scenario_a_quirk(X):
    X_s = X.copy()
    has_inc = X_s["MonthlyIncome"].notnull()
    X_s.loc[has_inc, "MonthlyIncome"] = X_s.loc[has_inc, "MonthlyIncome"] * 0.80
    X_s.loc[has_inc, "DebtRatio"]     = X_s.loc[has_inc, "DebtRatio"] / 0.80
    return X_s

# Scenario (b): DebtRatio x1.25 only
def get_scenario_b(X):
    X_s = X.copy()
    X_s["DebtRatio"] = X_s["DebtRatio"] * 1.25
    return X_s

# Scenario (c): Both combined (Original double-counted version: Income x0.80 AND DebtRatio x1.25)
def get_scenario_c(X):
    X_s = X.copy()
    X_s["MonthlyIncome"] = X_s["MonthlyIncome"] * 0.80
    X_s["DebtRatio"]     = X_s["DebtRatio"] * 1.25
    return X_s

# Scenario (e): Targeted shock: Income -20% to bottom 30% of earners
def get_scenario_e(X):
    X_s = X.copy()
    # Bottom 30% threshold among non-null incomes
    inc_30th = X_s["MonthlyIncome"].dropna().quantile(0.30)
    low_inc_mask = (X_s["MonthlyIncome"].notnull()) & (X_s["MonthlyIncome"] <= inc_30th)
    X_s.loc[low_inc_mask, "MonthlyIncome"] = X_s.loc[low_inc_mask, "MonthlyIncome"] * 0.80
    X_s.loc[low_inc_mask, "DebtRatio"]     = X_s.loc[low_inc_mask, "DebtRatio"] / 0.80
    return X_s

scenarios = {
    "(a) Income -20% (Recomputed DebtRatio +25%)": get_scenario_a,
    "(a*) Income -20% (Quirk-Aware: non-null only)": get_scenario_a_quirk,
    "(b) DebtRatio +25% Only (Constant Income)":   get_scenario_b,
    "(c) Original Combined (-20% Inc & +25% Debt)": get_scenario_c,
    "(e) Targeted Shock (-20% Inc to Bottom 30%)":  get_scenario_e,
}

scenario_rows = []
for sc_name, sc_func in scenarios.items():
    X_stressed = sc_func(X_test_base)
    for name, fname, color in MODEL_SPECS:
        b_info = baseline_metrics[name]
        m = b_info["model"]
        auc_base = b_info["auc_base"]
        mp_base  = b_info["mean_prob_base"]
        sd_base  = b_info["share_def_base"]
        t        = b_info["thresh"]

        p_stress = m.predict_proba(X_stressed)[:, 1]
        pred_stress = (p_stress >= t).astype(int)

        auc_stress = roc_auc_score(y_test, p_stress)
        retained_auc = (auc_stress / auc_base) * 100.0
        mp_stress = float(np.mean(p_stress))
        sd_stress = float(np.mean(pred_stress))

        pct_change_mean_prob = ((mp_stress - mp_base) / mp_base) * 100.0
        pct_change_share_def = ((sd_stress - sd_base) / sd_base) * 100.0 if sd_base > 0 else 0.0

        scenario_rows.append({
            "Scenario": sc_name,
            "Model": name,
            "Baseline_AUC": round(auc_base, 4),
            "Stressed_AUC": round(auc_stress, 4),
            "AUC_Retention_%": round(retained_auc, 3),
            "Baseline_Mean_Pred_Prob": round(mp_base, 4),
            "Stressed_Mean_Pred_Prob": round(mp_stress, 4),
            "Pct_Change_Mean_Pred_Probability": round(pct_change_mean_prob, 2),
            "Threshold_Used": round(t, 4),
            "Baseline_Pred_Default_Share": round(sd_base, 4),
            "Stressed_Pred_Default_Share": round(sd_stress, 4),
            "Pct_Change_Pred_Default_Share": round(pct_change_share_def, 2),
        })

sc_df = pd.DataFrame(scenario_rows)
sc_df.to_csv(os.path.join(TABLE_DIR, "stress_test_scenarios.csv"), index=False)
print("\n=== Stress Test Main Scenarios ===")
print(sc_df[["Scenario", "Model", "Baseline_AUC", "Stressed_AUC", "AUC_Retention_%", "Pct_Change_Mean_Pred_Probability", "Pct_Change_Pred_Default_Share"]].to_string())

# ---------------------------------------------------------------------------
# Scenario (d): Severity Sweep (-10%, -20%, -30%, -40%)
# ---------------------------------------------------------------------------
severity_levels = [0.10, 0.20, 0.30, 0.40]
severity_rows = []

for shock in severity_levels:
    shock_pct = int(shock * 100)
    inc_factor = 1.0 - shock
    debt_factor = 1.0 / inc_factor

    X_s = X_test_base.copy()
    X_s["MonthlyIncome"] = X_s["MonthlyIncome"] * inc_factor
    X_s["DebtRatio"]     = X_s["DebtRatio"] * debt_factor

    for name, fname, color in MODEL_SPECS:
        b_info = baseline_metrics[name]
        m = b_info["model"]
        auc_base = b_info["auc_base"]
        mp_base  = b_info["mean_prob_base"]
        sd_base  = b_info["share_def_base"]
        t        = b_info["thresh"]

        p_stress = m.predict_proba(X_s)[:, 1]
        pred_stress = (p_stress >= t).astype(int)

        auc_stress = roc_auc_score(y_test, p_stress)
        retained_auc = (auc_stress / auc_base) * 100.0
        mp_stress = float(np.mean(p_stress))
        sd_stress = float(np.mean(pred_stress))

        pct_change_mean_prob = ((mp_stress - mp_base) / mp_base) * 100.0
        pct_change_share_def = ((sd_stress - sd_base) / sd_base) * 100.0 if sd_base > 0 else 0.0

        severity_rows.append({
            "Income_Cut_%": f"-{shock_pct}%",
            "DebtRatio_Expansion_%": f"+{((debt_factor-1.0)*100):.1f}%",
            "Model": name,
            "Baseline_AUC": round(auc_base, 4),
            "Stressed_AUC": round(auc_stress, 4),
            "AUC_Retention_%": round(retained_auc, 3),
            "Pct_Change_Mean_Pred_Probability": round(pct_change_mean_prob, 2),
            "Pct_Change_Pred_Default_Share": round(pct_change_share_def, 2),
        })

sev_df = pd.DataFrame(severity_rows)
sev_df.to_csv(os.path.join(TABLE_DIR, "stress_test_severity_sweep.csv"), index=False)
print("\n=== Stress Test Severity Sweep ===")
print(sev_df.to_string())

# ---------------------------------------------------------------------------
# Visualization: Publication-grade Figure 6 (Multi-Panel Stress Test)
# ---------------------------------------------------------------------------
plt.rcParams.update({"font.size": 8.5, "font.family": "serif", "figure.dpi": 300})
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8))

# Left panel: Sensitivity by Model under Main Recomputed Scenario (a)
df_sc_a = sc_df[sc_df["Scenario"] == "(a) Income -20% (Recomputed DebtRatio +25%)"]
models_order = [m[0] for m in MODEL_SPECS]
df_sc_a = df_sc_a.set_index("Model").loc[models_order].reset_index()

x = np.arange(len(models_order))
colors = [m[2] for m in MODEL_SPECS]

bars = axes[0].bar(x, df_sc_a["Pct_Change_Mean_Pred_Probability"], color=colors, width=0.55, edgecolor="black", linewidth=0.5)
axes[0].set_ylabel("% Change in Mean Predicted Default Probability", fontsize=8.5)
axes[0].set_title("(A) Probability Sensitivity under -20% Income Shock", fontsize=9.5, fontweight="bold")
axes[0].set_xticks(x)
axes[0].set_xticklabels(models_order, rotation=25, ha="right", fontsize=7.5)
axes[0].grid(True, linestyle=":", alpha=0.5, axis="y")

for bar in bars:
    h = bar.get_height()
    axes[0].annotate(f"{h:+.2f}%", (bar.get_x() + bar.get_width() / 2., h),
                     ha="center", va="bottom" if h >= 0 else "top", fontsize=7, xytext=(0, 2), textcoords="offset points")

# Right panel: Severity Sweep Trajectory for Top Models
top_models = ["LightGBM", "XGBoost", "Random Forest", "Logistic Regression", "SVM"]
shock_labels = [-10, -20, -30, -40]
for name in top_models:
    sub = sev_df[sev_df["Model"] == name]
    y_vals = sub["Pct_Change_Mean_Pred_Probability"].values
    c = [m[2] for m in MODEL_SPECS if m[0] == name][0]
    axes[1].plot(shock_labels, y_vals, marker="o", markersize=4, label=name, color=c, linewidth=1.5)

axes[1].set_xlabel("Simulated Exogenous Income Reduction (%)", fontsize=8.5)
axes[1].set_ylabel("% Change in Mean Predicted Probability", fontsize=8.5)
axes[1].set_title("(B) Shock Severity Response Trajectory", fontsize=9.5, fontweight="bold")
axes[1].grid(True, linestyle=":", alpha=0.5)
axes[1].legend(loc="upper left", frameon=True, fontsize=7.5)

plt.tight_layout()
fig6_png = os.path.join(FIG_DIR, "fig6_stress_test_v2.png")
fig6_pdf = os.path.join(FIG_DIR, "fig6_stress_test_v2.pdf")
plt.savefig(fig6_png, dpi=300)
plt.savefig(fig6_pdf)
plt.close()
print(f"\nSaved Fig 6: {fig6_png} and {fig6_pdf}")

print("\nStep 4 Complete!")
