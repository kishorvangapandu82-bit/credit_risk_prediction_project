"""
=============================================================================
PHASE 10B -- MACROECONOMIC STRESS TESTING
=============================================================================
Project : GMSC Credit Scoring Research
File    : src/stress_testing.py
Purpose : Subject all 4 trained models to a simulated macroeconomic shock:
          - MonthlyIncome falls by 20% (STRESS_INCOME_FACTOR = 0.80)
          - DebtRatio rises by 25%     (STRESS_DEBTRATIO_FACTOR = 1.25)
          
Assess:
  1. Model ranking stability under covariate shift
  2. ROC-AUC degradation / robustness
  3. Upward migration of predicted default probabilities
  4. Percentage increase in predicted defaults (capital adequacy implications)

Outputs:
  - results/stress_test_comparison.csv
  - plots/stress_test_impact.png
=============================================================================
RUN COMMAND:
  cd credit-scoring-project
  python src/stress_testing.py
=============================================================================
"""

import sys, os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
from config import (
    MODEL_DIR, RESULT_DIR, PLOT_DIR,
    STRESS_INCOME_FACTOR, STRESS_DEBTRATIO_FACTOR
)
from data_preprocessing import load_data, clean_data, split_data


MODELS = [
    {"name": "Logistic Regression",    "file": "logistic_regression.joblib", "color": "#1f77b4"},
    {"name": "Decision Tree",          "file": "decision_tree.joblib",       "color": "#ff7f0e"},
    {"name": "MLP Neural Network",     "file": "mlp.joblib",                 "color": "#9467bd"},
    {"name": "XGBoost",                "file": "xgboost.joblib",             "color": "#2ca02c"},
    {"name": "LightGBM",               "file": "lightgbm.joblib",            "color": "#17becf"},
    {"name": "Support Vector Machine", "file": "svm.joblib",                 "color": "#d62728"},
    {"name": "Random Forest",          "file": "random_forest.joblib",       "color": "#8c564b"},
]


def run_stress_test():
    print("=" * 65)
    print("PHASE 10B -- MACROECONOMIC STRESS TESTING")
    print("=" * 65)
    print(f"\n  Macroeconomic Shock Parameters:")
    print(f"    - MonthlyIncome factor  : {STRESS_INCOME_FACTOR:.2f} (-20% income reduction)")
    print(f"    - DebtRatio factor      : {STRESS_DEBTRATIO_FACTOR:.2f} (+25% debt service increase)")

    # 1. Load data
    df_raw = load_data()
    df_clean, _ = clean_data(df_raw, verbose=False)
    _, X_test_base, _, y_test = split_data(df_clean)

    # 2. Create stressed test dataset
    X_test_stressed = X_test_base.copy()
    X_test_stressed["MonthlyIncome"] = X_test_stressed["MonthlyIncome"] * STRESS_INCOME_FACTOR
    X_test_stressed["DebtRatio"]     = X_test_stressed["DebtRatio"] * STRESS_DEBTRATIO_FACTOR

    print(f"\n  Baseline Test Set Shape : {X_test_base.shape}")
    print(f"  Stressed Test Set Shape : {X_test_stressed.shape}")
    print(f"  Baseline MonthlyIncome Mean: ${X_test_base['MonthlyIncome'].dropna().mean():,.2f}")
    print(f"  Stressed MonthlyIncome Mean: ${X_test_stressed['MonthlyIncome'].dropna().mean():,.2f}")
    print(f"  Baseline DebtRatio Mean    : {X_test_base['DebtRatio'].mean():.4f}")
    print(f"  Stressed DebtRatio Mean    : {X_test_stressed['DebtRatio'].mean():.4f}")

    results = []
    prob_distributions = {}

    print("\n[Step 2] Evaluating models under Baseline vs. Stressed conditions...")
    for m in MODELS:
        m_path = os.path.join(MODEL_DIR, m["file"])
        model = joblib.load(m_path)

        # Baseline predictions
        y_prob_base = model.predict_proba(X_test_base)[:, 1]
        y_pred_base = model.predict(X_test_base)
        auc_base    = roc_auc_score(y_test, y_prob_base)
        def_rate_base = y_pred_base.mean()

        # Stressed predictions
        y_prob_stress = model.predict_proba(X_test_stressed)[:, 1]
        y_pred_stress = model.predict(X_test_stressed)
        auc_stress    = roc_auc_score(y_test, y_prob_stress)
        def_rate_stress = y_pred_stress.mean()

        # Metrics delta
        auc_drop = auc_base - auc_stress
        def_rate_increase = ((def_rate_stress - def_rate_base) / def_rate_base) * 100 if def_rate_base > 0 else 0
        mean_prob_base = y_prob_base.mean()
        mean_prob_stress = y_prob_stress.mean()

        prob_distributions[m["name"]] = {
            "base": y_prob_base,
            "stress": y_prob_stress,
            "color": m["color"]
        }

        records = {
            "Model": m["name"],
            "Baseline_AUC": round(auc_base, 4),
            "Stressed_AUC": round(auc_stress, 4),
            "AUC_Drop": round(auc_drop, 4),
            "AUC_Retention_%": round((auc_stress / auc_base) * 100, 2),
            "Baseline_Mean_Prob": round(mean_prob_base, 4),
            "Stressed_Mean_Prob": round(mean_prob_stress, 4),
            "Prob_Increase_%": round(((mean_prob_stress - mean_prob_base) / mean_prob_base) * 100, 2),
            "Baseline_Pred_Default_Rate": round(def_rate_base, 4),
            "Stressed_Pred_Default_Rate": round(def_rate_stress, 4),
            "Default_Rate_Surge_%": round(def_rate_increase, 2),
        }
        results.append(records)

    results_df = pd.DataFrame(results)

    # 3. Print results table
    print("\n" + "=" * 80)
    print("STRESS TEST RESULTS: BASELINE VS. STRESSED MACROECONOMIC SCENARIO")
    print("=" * 80)
    print(f"  {'Model':<22} {'Base AUC':>10} {'Stress AUC':>12} {'AUC Drop':>10} {'Mean Prob Delta':>16} {'Def Surge':>12}")
    print("  " + "-" * 80)
    for _, r in results_df.iterrows():
        print(f"  {r['Model']:<22} {r['Baseline_AUC']:>10.4f} {r['Stressed_AUC']:>12.4f} {r['AUC_Drop']:>+10.4f} {r['Prob_Increase_%']:>+13.2f}% {r['Default_Rate_Surge_%']:>+11.2f}%")

    # 4. Save results to CSV
    os.makedirs(RESULT_DIR, exist_ok=True)
    out_csv = os.path.join(RESULT_DIR, "stress_test_comparison.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\n[Step 3] Results saved to: {out_csv}")

    # 5. Generate Stress Test Plot
    print("\n[Step 4] Generating Stress Test Visualizations...")
    os.makedirs(PLOT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid")

    # Left plot: AUC Degradation
    x = np.arange(len(results_df))
    width = 0.35

    ax1 = axes[0]
    rects1 = ax1.bar(x - width/2, results_df["Baseline_AUC"], width, label="Baseline", color="#4a90e2")
    rects2 = ax1.bar(x + width/2, results_df["Stressed_AUC"], width, label="Stressed (-20% Inc, +25% Debt)", color="#e94e77")

    ax1.set_ylabel("ROC-AUC", fontsize=12, labelpad=8)
    ax1.set_title("Model Discriminative Power: Baseline vs. Stressed", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(results_df["Model"], rotation=15, fontsize=10)
    ax1.set_ylim([0.70, 0.90])
    ax1.legend(loc="lower right", frameon=True, facecolor="white")

    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f"{h:.3f}", (rect.get_x() + rect.get_width()/2., h),
                     ha="center", va="bottom", fontsize=9, xytext=(0, 2), textcoords="offset points")
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f"{h:.3f}", (rect.get_x() + rect.get_width()/2., h),
                     ha="center", va="bottom", fontsize=9, xytext=(0, 2), textcoords="offset points")

    # Right plot: Mean Default Probability Surge
    ax2 = axes[1]
    prob_bars = ax2.bar(results_df["Model"], results_df["Prob_Increase_%"], color="#e67e22", width=0.5)
    ax2.set_ylabel("% Increase in Mean Default Probability", fontsize=12, labelpad=8)
    ax2.set_title("Risk Sensitivity: Default Probability Surge under Shock", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(results_df["Model"], rotation=15, fontsize=10)
    ax2.set_ylim([0, max(results_df["Prob_Increase_%"]) * 1.25])

    for rect in prob_bars:
        h = rect.get_height()
        ax2.annotate(f"+{h:.1f}%", (rect.get_x() + rect.get_width()/2., h),
                     ha="center", va="bottom", fontsize=9.5, fontweight="bold", xytext=(0, 3), textcoords="offset points")

    plt.suptitle("Macroeconomic Stress Testing Analysis (-20% Income, +25% DebtRatio)", fontsize=15, fontweight="bold", y=1.00)
    plt.tight_layout()

    out_plot = os.path.join(PLOT_DIR, "stress_test_impact.png")
    plt.savefig(out_plot)
    plt.close()
    print(f"  Saved plot: {out_plot}")

    print("\n" + "=" * 65)
    print("PHASE 10B COMPLETE -- STRESS TESTING FINISHED")
    print("=" * 65)


if __name__ == "__main__":
    run_stress_test()
