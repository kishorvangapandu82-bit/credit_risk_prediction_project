"""
=============================================================================
V2 PIPELINE -- STEP 6: SHAP GAME-THEORETIC EXPLAINABILITY AUDIT
=============================================================================
1. TreeSHAP computation for baseline LightGBM model on holdout test partition.
2. Generates:
   - Global Beeswarm summary plot (PDF + PNG 300 dpi)
   - Global Mean |SHAP| feature ranking bar chart (PDF + PNG 300 dpi)
   - Partial dependence plots for the top 4 most influential features (PDF + PNG 300 dpi)
   - Local waterfall attribution plots for a High-Risk and Low-Risk borrower (PDF + PNG 300 dpi)
3. Computes TreeSHAP for baseline XGBoost model.
4. Evaluates Spearman rank agreement between LightGBM and XGBoost feature attributions.
5. Saves outputs_v2/tables/shap_rankings.csv.
=============================================================================
"""

import os, sys, joblib, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shap

# Add src to path
sys.path.insert(0, os.path.abspath("src"))
from config import RANDOM_STATE, FEATURES
from data_preprocessing import load_data, clean_data, split_data

TABLE_DIR = "outputs_v2/tables"
FIG_DIR   = "outputs_v2/figures"
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# 1. Load data
print("[SHAP] Loading holdout test partition...")
df_raw = load_data()
df_clean, _ = clean_data(df_raw, verbose=False)
_, X_test, _, y_test = split_data(df_clean)

# Baseline models
lgb_model = joblib.load("models/lightgbm.joblib")
xgb_model = joblib.load("models/xgboost.joblib")

# Transform test features using the pipeline's fitted preprocessor
preprocessor = lgb_model.named_steps["preprocessor"]
X_test_trans = preprocessor.transform(X_test)

# Extract correct transformed feature names from ColumnTransformer
raw_names_out = preprocessor.get_feature_names_out()
TRANSFORMED_FEATURES = [f.split("__")[-1] for f in raw_names_out]
X_test_df = pd.DataFrame(X_test_trans, columns=TRANSFORMED_FEATURES, index=X_test.index)

# Sample N=2,000 stratified observations for SHAP calculation
SHAP_SAMPLE_SIZE = 2000
np.random.seed(RANDOM_STATE)
sample_indices = np.random.choice(len(X_test_df), size=SHAP_SAMPLE_SIZE, replace=False)
X_sample = X_test_df.iloc[sample_indices]
y_sample = y_test.iloc[sample_indices]

print(f"[SHAP] Computing TreeSHAP for LightGBM on {SHAP_SAMPLE_SIZE} test profiles...")
lgb_clf = lgb_model.named_steps["classifier"]
lgb_explainer = shap.TreeExplainer(lgb_clf)
lgb_shap_values = lgb_explainer(X_sample)

# Handle binary classification output format if needed
if len(lgb_shap_values.values.shape) == 3:
    lgb_vals = lgb_shap_values.values[:, :, 1]
    lgb_base_val = lgb_shap_values.base_values[0, 1]
else:
    lgb_vals = lgb_shap_values.values
    lgb_base_val = lgb_shap_values.base_values[0] if isinstance(lgb_shap_values.base_values, (np.ndarray, list)) else lgb_shap_values.base_values

# Mean Absolute SHAP values for LightGBM
lgb_mean_abs = np.mean(np.abs(lgb_vals), axis=0)
lgb_rank_df = pd.DataFrame({
    "Feature": TRANSFORMED_FEATURES,
    "LightGBM_Mean_Abs_SHAP": lgb_mean_abs
}).sort_values("LightGBM_Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
lgb_rank_df["LightGBM_Rank"] = lgb_rank_df.index + 1

# TreeSHAP for XGBoost
print(f"[SHAP] Computing TreeSHAP for XGBoost on {SHAP_SAMPLE_SIZE} test profiles...")
xgb_clf = xgb_model.named_steps["classifier"]
xgb_explainer = shap.TreeExplainer(xgb_clf)
xgb_shap_values = xgb_explainer(X_sample)

if len(xgb_shap_values.values.shape) == 3:
    xgb_vals = xgb_shap_values.values[:, :, 1]
else:
    xgb_vals = xgb_shap_values.values

xgb_mean_abs = np.mean(np.abs(xgb_vals), axis=0)
xgb_rank_df = pd.DataFrame({
    "Feature": TRANSFORMED_FEATURES,
    "XGBoost_Mean_Abs_SHAP": xgb_mean_abs
}).sort_values("XGBoost_Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
xgb_rank_df["XGBoost_Rank"] = xgb_rank_df.index + 1

# Merge rankings
shap_merged = pd.merge(lgb_rank_df, xgb_rank_df, on="Feature")
shap_merged = shap_merged.sort_values("LightGBM_Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

# Spearman rank correlation
corr, p_val = spearmanr(shap_merged["LightGBM_Mean_Abs_SHAP"], shap_merged["XGBoost_Mean_Abs_SHAP"])
print(f"\nSpearman Rank Correlation (LightGBM vs XGBoost SHAP): {corr:.4f} (p-value: {p_val:.4e})")

shap_merged["Spearman_Rho"] = round(corr, 4)
shap_merged["Spearman_p_value"] = round(p_val, 6)
shap_merged.to_csv(os.path.join(TABLE_DIR, "shap_rankings.csv"), index=False)
print("Saved outputs_v2/tables/shap_rankings.csv")

# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------
plt.rcParams.update({"font.size": 8.5, "font.family": "serif", "figure.dpi": 300})

# 1. Beeswarm Plot (Fig 7)
fig = plt.figure(figsize=(7.0, 4.5))
shap.summary_plot(lgb_vals, X_sample, feature_names=TRANSFORMED_FEATURES, show=False, plot_size=(7.0, 4.5))
plt.title("TreeSHAP Beeswarm Summary Plot (Baseline LightGBM)", fontsize=10.5, fontweight="bold", pad=12)
plt.xlabel("SHAP Value (Impact on Log-Odds of Default)", fontsize=9)
plt.tight_layout()
bw_png = os.path.join(FIG_DIR, "fig7_shap_summary_beeswarm_v2.png")
bw_pdf = os.path.join(FIG_DIR, "fig7_shap_summary_beeswarm_v2.pdf")
plt.savefig(bw_png, dpi=300)
plt.savefig(bw_pdf)
plt.close()
print(f"Saved: {bw_png} and {bw_pdf}")

# 2. Mean |SHAP| Bar Chart (Fig 8)
fig, ax = plt.subplots(figsize=(6.5, 4.0))
y_pos = np.arange(len(shap_merged))
ax.barh(y_pos, shap_merged["LightGBM_Mean_Abs_SHAP"], color="#17becf", edgecolor="black", linewidth=0.5, height=0.6)
ax.set_yticks(y_pos)
ax.set_yticklabels(shap_merged["Feature"], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Mean Absolute SHAP Value (E[|SHAP|])", fontsize=9)
ax.set_title("Global Feature Importance (TreeSHAP Mean Attribution)", fontsize=10.5, fontweight="bold")
ax.grid(True, linestyle=":", alpha=0.5, axis="x")

for i, v in enumerate(shap_merged["LightGBM_Mean_Abs_SHAP"]):
    ax.text(v + 0.01, i, f"{v:.4f}", va="center", fontsize=7.5)

plt.tight_layout()
bar_png = os.path.join(FIG_DIR, "fig8_shap_bar_chart_v2.png")
bar_pdf = os.path.join(FIG_DIR, "fig8_shap_bar_chart_v2.pdf")
plt.savefig(bar_png, dpi=300)
plt.savefig(bar_pdf)
plt.close()
print(f"Saved: {bar_png} and {bar_pdf}")

# 3. Dependence Plots for Top 4 Features
top_4_feats = shap_merged["Feature"].head(4).tolist()
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.5))
axes = axes.flatten()

for idx, feat in enumerate(top_4_feats):
    feat_idx = TRANSFORMED_FEATURES.index(feat)
    x_val = X_sample[feat]
    s_val = lgb_vals[:, feat_idx]
    
    axes[idx].scatter(x_val, s_val, alpha=0.35, color="#1f77b4", s=10, edgecolors="none")
    axes[idx].set_xlabel(feat, fontsize=8)
    axes[idx].set_ylabel("SHAP Value", fontsize=8)
    axes[idx].set_title(f"Dependence: {feat}", fontsize=9, fontweight="bold")
    axes[idx].grid(True, linestyle=":", alpha=0.4)

plt.tight_layout()
dep_png = os.path.join(FIG_DIR, "shap_dependence_top4_v2.png")
dep_pdf = os.path.join(FIG_DIR, "shap_dependence_top4_v2.pdf")
plt.savefig(dep_png, dpi=300)
plt.savefig(dep_pdf)
plt.close()
print(f"Saved: {dep_png} and {dep_pdf}")

# 4. Local Waterfall Plots (High-Risk vs Low-Risk Borrowers)
sample_probs = lgb_model.predict_proba(X_test.iloc[sample_indices])[:, 1]
high_risk_idx = np.argmax(sample_probs)
low_risk_idx  = np.argmin(sample_probs)

expl_high = shap.Explanation(
    values=lgb_vals[high_risk_idx],
    base_values=lgb_base_val,
    data=X_sample.iloc[high_risk_idx].values,
    feature_names=TRANSFORMED_FEATURES
)

expl_low = shap.Explanation(
    values=lgb_vals[low_risk_idx],
    base_values=lgb_base_val,
    data=X_sample.iloc[low_risk_idx].values,
    feature_names=TRANSFORMED_FEATURES
)

# Waterfall High Risk
fig = plt.figure(figsize=(6.5, 4.5))
shap.plots.waterfall(expl_high, max_display=8, show=False)
plt.title(f"Local Adverse Action Waterfall: High-Risk Borrower (p = {sample_probs[high_risk_idx]:.3f})", fontsize=9.5, fontweight="bold", pad=10)
plt.tight_layout()
wf_hi_png = os.path.join(FIG_DIR, "shap_waterfall_high_risk_v2.png")
wf_hi_pdf = os.path.join(FIG_DIR, "shap_waterfall_high_risk_v2.pdf")
plt.savefig(wf_hi_png, dpi=300)
plt.savefig(wf_hi_pdf)
plt.close()

# Waterfall Low Risk
fig = plt.figure(figsize=(6.5, 4.5))
shap.plots.waterfall(expl_low, max_display=8, show=False)
plt.title(f"Local Adverse Action Waterfall: Low-Risk Borrower (p = {sample_probs[low_risk_idx]:.3f})", fontsize=9.5, fontweight="bold", pad=10)
plt.tight_layout()
wf_lo_png = os.path.join(FIG_DIR, "shap_waterfall_low_risk_v2.png")
wf_lo_pdf = os.path.join(FIG_DIR, "shap_waterfall_low_risk_v2.pdf")
plt.savefig(wf_lo_png, dpi=300)
plt.savefig(wf_lo_pdf)
plt.close()
print("Saved local waterfall plots!")

print("\nStep 6 Complete!")
