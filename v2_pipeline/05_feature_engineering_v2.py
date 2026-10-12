"""
=============================================================================
V2 PIPELINE -- STEP 5: DOMAIN-SPECIFIC FINANCIAL FEATURE ENGINEERING
=============================================================================
1. Generates outputs_v2/tables/engineered_features.csv:
   - Feature Name, Mathematical Formula, Economic / Behavioral Rationale,
     and Anti-Leakage Confirmation (Training-Only Stats).
2. Evaluates 10-Feature vs 18-Feature models on holdout test set for:
   - Random Forest
   - XGBoost
   - LightGBM
3. Computes 1,000-sample paired bootstrap Confidence Intervals on the
   AUC change (Delta AUC = AUC_18 - AUC_10).
4. Generates ROC comparison curve in PDF + PNG 300 dpi.
=============================================================================
"""

import os, sys, joblib, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, roc_curve

# Add src to path
sys.path.insert(0, os.path.abspath("src"))
from config import RANDOM_STATE
from data_preprocessing import load_data, clean_data, split_data
from feature_engineering import engineer_features, ALL_ENGINEERED_FEATURES
from models import build_random_forest_pipeline, build_xgboost_pipeline, build_lightgbm_pipeline

TABLE_DIR = "outputs_v2/tables"
FIG_DIR   = "outputs_v2/figures"
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# 1. Document the 8 engineered features in CSV
features_meta = [
    {
        "Feature_Name": "TotalPastDueCount",
        "Mathematical_Formula": "NumberOfTime30-59Days + NumberOfTime60-89Days + NumberOfTimes90DaysLate",
        "Economic_Rationale": "Aggregates overall delinquency incidence across all temporal severity buckets to capture cumulative distress frequency.",
        "Leakage_Protocol": "Strictly row-wise aggregation; no cross-observation statistics utilized."
    },
    {
        "Feature_Name": "DelinquencySeverityIndex",
        "Mathematical_Formula": "1.0*(30-59Days) + 2.0*(60-89Days) + 3.0*(90+DaysLate)",
        "Economic_Rationale": "Weights severe default signals linearly higher (90+ days weighted 3x vs 30-day delays), reflecting ordinal risk escalation.",
        "Leakage_Protocol": "Strictly row-wise linear combination; zero cross-observation leakage."
    },
    {
        "Feature_Name": "IncomePerDependent",
        "Mathematical_Formula": "MonthlyIncome_imputed / (NumberOfDependents_imputed + 1.0)",
        "Economic_Rationale": "Normalizes gross household purchasing power per dependent member to evaluate per-capita subsistence margin.",
        "Leakage_Protocol": "Imputation median and mode computed strictly on the training partition."
    },
    {
        "Feature_Name": "EstimatedMonthlyDebt",
        "Mathematical_Formula": "MonthlyIncome_imputed * clip(DebtRatio, upper=10.0)",
        "Economic_Rationale": "Converts dimensionless debt ratio into estimated dollar-denominated monthly debt service obligations.",
        "Leakage_Protocol": "Income median fit on training data; ratio clipping bound is constant."
    },
    {
        "Feature_Name": "DisposableIncome",
        "Mathematical_Formula": "MonthlyIncome_imputed - EstimatedMonthlyDebt",
        "Economic_Rationale": "Estimates free residual cash flow buffer available to absorb emergency financial shocks.",
        "Leakage_Protocol": "Derived from training-imputed income and debt estimates; zero test leakage."
    },
    {
        "Feature_Name": "UnsecuredLinesRatio",
        "Mathematical_Formula": "(NumberOfOpenCreditLines - NumberRealEstateLoans) / (NumberOfOpenCreditLines + 1.0)",
        "Economic_Rationale": "Isolates uncollateralized revolving credit cards from secured real estate mortgages to measure debt vulnerability.",
        "Leakage_Protocol": "Strictly row-wise operation on applicant loan counts."
    },
    {
        "Feature_Name": "MissingIncomeFlag",
        "Mathematical_Formula": "Indicator(MonthlyIncome is NaN)",
        "Economic_Rationale": "Flags unverified or undocumented income applications, which strongly correlate with non-traditional or high-risk employment.",
        "Leakage_Protocol": "Identified prior to imputation on raw applicant entries."
    },
    {
        "Feature_Name": "OverUtilizationFlag",
        "Mathematical_Formula": "Indicator(RevolvingUtilizationOfUnsecuredLines > 1.0)",
        "Economic_Rationale": "Identifies applicants currently exceeding their total approved credit limit, a primary distress indicator.",
        "Leakage_Protocol": "Row-wise threshold comparison; completely leakage-free."
    },
]

feat_df = pd.DataFrame(features_meta)
feat_df.to_csv(os.path.join(TABLE_DIR, "engineered_features.csv"), index=False)
print("Saved engineered features documentation to outputs_v2/tables/engineered_features.csv")

# 2. Prepare 10-feature and 18-feature datasets
print("\n[Feature Engineering] Preparing datasets...")
df_raw = load_data()
df_clean, _ = clean_data(df_raw, verbose=False)
X_train_10, X_test_10, y_train, y_test = split_data(df_clean)

# Engineer features using training-set median strictly
X_train_18, inc_median = engineer_features(X_train_10)
X_test_18, _          = engineer_features(X_test_10, income_median=inc_median)

print(f"  10-Feature train shape: {X_train_10.shape}, test shape: {X_test_10.shape}")
print(f"  18-Feature train shape: {X_train_18.shape}, test shape: {X_test_18.shape}")

# Models to evaluate
ENG_MODELS = [
    ("Random Forest", "random_forest.joblib",
     lambda: build_random_forest_pipeline(n_estimators=200, max_depth=12, min_samples_split=50)),
    ("XGBoost", "xgboost.joblib",
     lambda: build_xgboost_pipeline(learning_rate=0.05, max_depth=5, n_estimators=200, scale_pos_weight=13.96)),
    ("LightGBM", "lightgbm.joblib",
     lambda: build_lightgbm_pipeline(learning_rate=0.05, num_leaves=31, n_estimators=200, scale_pos_weight=13.96)),
]

y_test_arr = y_test.values
N_BOOTSTRAP = 1000
np.random.seed(RANDOM_STATE)
boot_indices = [np.random.choice(len(y_test_arr), size=len(y_test_arr), replace=True) for _ in range(N_BOOTSTRAP)]

comparison_rows = []
roc_curves_data = {}

for name, base_fname, pipe_builder in ENG_MODELS:
    print(f"\nEvaluating {name}...")
    # Baseline 10-feature model
    m_10 = joblib.load(os.path.join("models", base_fname))
    p_10 = m_10.predict_proba(X_test_10)[:, 1]
    pred_10 = m_10.predict(X_test_10)
    auc_10 = roc_auc_score(y_test, p_10)
    rec_10 = recall_score(y_test, pred_10)
    prec_10 = precision_score(y_test, pred_10)
    f1_10 = f1_score(y_test, pred_10)
    acc_10 = accuracy_score(y_test, pred_10)

    # Train 18-feature model
    print(f"  Fitting {name} on 18 engineered features...")
    pipe_18 = pipe_builder()
    pipe_18.fit(X_train_18, y_train)
    p_18 = pipe_18.predict_proba(X_test_18)[:, 1]
    pred_18 = pipe_18.predict(X_test_18)
    auc_18 = roc_auc_score(y_test, p_18)
    rec_18 = recall_score(y_test, pred_18)
    prec_18 = precision_score(y_test, pred_18)
    f1_18 = f1_score(y_test, pred_18)
    acc_18 = accuracy_score(y_test, pred_18)

    delta_auc = auc_18 - auc_10

    # Paired Bootstrap for Delta AUC
    delta_boots = np.zeros(N_BOOTSTRAP)
    for b_idx, idx in enumerate(boot_indices):
        y_b = y_test_arr[idx]
        if len(np.unique(y_b)) < 2:
            continue
        auc_b_10 = roc_auc_score(y_b, p_10[idx])
        auc_b_18 = roc_auc_score(y_b, p_18[idx])
        delta_boots[b_idx] = auc_b_18 - auc_b_10

    ci_low = np.percentile(delta_boots, 2.5)
    ci_high = np.percentile(delta_boots, 97.5)
    p_val = 2.0 * min(np.mean(delta_boots <= 0), np.mean(delta_boots >= 0))
    p_val = min(max(p_val, 1.0 / N_BOOTSTRAP), 1.0)

    roc_curves_data[name] = {
        "p_10": p_10,
        "p_18": p_18,
        "auc_10": auc_10,
        "auc_18": auc_18,
    }

    comparison_rows.append({
        "Model": name,
        "Features_10_AUC": round(auc_10, 4),
        "Features_18_AUC": round(auc_18, 4),
        "Delta_AUC": round(delta_auc, 4),
        "Delta_AUC_95_CI": f"[{ci_low:.4f}, {ci_high:.4f}]",
        "Bootstrap_p_value": round(p_val, 4),
        "Features_10_F1": round(f1_10, 4),
        "Features_18_F1": round(f1_18, 4),
        "Features_10_Precision": round(prec_10, 4),
        "Features_18_Precision": round(prec_18, 4),
        "Features_10_Recall": round(rec_10, 4),
        "Features_18_Recall": round(rec_18, 4),
        "Features_10_Accuracy": round(acc_10, 4),
        "Features_18_Accuracy": round(acc_18, 4),
    })

comp_df = pd.DataFrame(comparison_rows)
comp_df.to_csv(os.path.join(TABLE_DIR, "feature_engineering_comparison.csv"), index=False)
print("\n=== Feature Engineering (10 vs 18 Features) Comparison ===")
print(comp_df.to_string())

# Plot ROC curves comparison
plt.rcParams.update({"font.size": 8.5, "font.family": "serif", "figure.dpi": 300})
fig, ax = plt.subplots(figsize=(5.5, 4.5))

colors = {"LightGBM": "#17becf", "XGBoost": "#2ca02c", "Random Forest": "#8c564b"}
for name, data in roc_curves_data.items():
    fpr10, tpr10, _ = roc_curve(y_test, data["p_10"])
    fpr18, tpr18, _ = roc_curve(y_test, data["p_18"])
    c = colors[name]
    ax.plot(fpr10, tpr10, linestyle="--", color=c, alpha=0.7, label=f"{name} (10 Feat, AUC={data['auc_10']:.4f})")
    ax.plot(fpr18, tpr18, linestyle="-",  color=c, linewidth=1.5, label=f"{name} (18 Feat, AUC={data['auc_18']:.4f})")

ax.plot([0, 1], [0, 1], "k:", alpha=0.5)
ax.set_xlabel("False Positive Rate", fontsize=9)
ax.set_ylabel("True Positive Rate", fontsize=9)
ax.set_title("ROC Comparison: 10 Baseline vs. 18 Engineered Features", fontsize=10, fontweight="bold")
ax.grid(True, linestyle=":", alpha=0.5)
ax.legend(loc="lower right", frameon=True, fontsize=7.5)
plt.tight_layout()

fe_png = os.path.join(FIG_DIR, "feature_engineering_roc_v2.png")
fe_pdf = os.path.join(FIG_DIR, "feature_engineering_roc_v2.pdf")
plt.savefig(fe_png, dpi=300)
plt.savefig(fe_pdf)
plt.close()
print(f"\nSaved plots: {fe_png} and {fe_pdf}")
print("\nStep 5 Complete!")
