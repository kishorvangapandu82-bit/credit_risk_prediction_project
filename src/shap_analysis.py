"""
=============================================================================
PHASE 11 -- SHAP (SHAPLEY ADDITIVE EXPLANATIONS) ANALYSIS
=============================================================================
Project : GMSC Credit Scoring Research
File    : src/shap_analysis.py
Purpose : Provide rigorous Explainable AI (XAI) for our top model (XGBoost):
          1. Global feature importance (Mean |SHAP|)
          2. Beeswarm summary plot (directionality of risk factors)
          3. Non-linear dependence plot (Revolving Utilization threshold)
          4. Individual local explanation (Waterfall plot for adverse action)
          5. Individual local explanation for low-risk borrower

Compliance Context:
  - Banking regulations (US ECOA / Fair Lending / FCRA / GDPR) require lenders
    to provide specific Adverse Action reasons when a loan is rejected.
  - SHAP translates complex black-box gradient boosting trees into compliant,
    exact credit risk drivers.

Outputs:
  - plots/shap_summary_beeswarm.png
  - plots/shap_feature_importance_bar.png
  - plots/shap_dependence_revolving_util.png
  - plots/shap_waterfall_high_risk.png
  - plots/shap_waterfall_low_risk.png
  - results/shap_feature_importance.csv
=============================================================================
RUN COMMAND:
  cd credit-scoring-project
  python src/shap_analysis.py
=============================================================================
"""

import sys, os
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MODEL_DIR, RESULT_DIR, PLOT_DIR, RANDOM_STATE
from data_preprocessing import load_data, clean_data, split_data


def run_shap_analysis():
    print("=" * 65)
    print("PHASE 11 -- SHAP EXPLAINABLE AI (XAI) ANALYSIS")
    print("=" * 65)

    # 1. Load pipeline and test data
    print("\n[Step 1] Loading XGBoost pipeline and holdout test data...")
    pipeline_path = os.path.join(MODEL_DIR, "xgboost.joblib")
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"Trained model not found at {pipeline_path}")
    
    pipeline = joblib.load(pipeline_path)
    preprocessor = pipeline.named_steps["preprocessor"]
    xgb_model    = pipeline.named_steps["classifier"]

    df_raw = load_data()
    df_clean, _ = clean_data(df_raw, verbose=False)
    _, X_test, _, y_test = split_data(df_clean)

    # Clean feature names (strip transformer prefixes)
    raw_feature_names = preprocessor.get_feature_names_out()
    feature_names = [f.split("__")[-1] for f in raw_feature_names]

    # Preprocess test samples
    # Sample 3,000 holdout observations for robust SHAP estimation
    sample_size = 3000
    np.random.seed(RANDOM_STATE)
    sample_idx = np.random.choice(len(X_test), size=sample_size, replace=False)
    
    X_sample_raw = X_test.iloc[sample_idx]
    y_sample     = y_test.iloc[sample_idx]
    
    X_sample_trans = preprocessor.transform(X_sample_raw)
    X_sample_df = pd.DataFrame(X_sample_trans, columns=feature_names, index=X_sample_raw.index)

    # 2. Compute TreeSHAP values
    print(f"\n[Step 2] Computing TreeSHAP values on {sample_size:,} holdout samples...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X_sample_df)
    shap_values.feature_names = feature_names

    # 3. Quantitative SHAP ranking table
    print("\n[Step 3] Calculating Mean |SHAP| feature importance table...")
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Abs_SHAP": mean_abs_shap
    }).sort_values("Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

    print("\n  SHAP Feature Importance Ranking:")
    print("  " + "-" * 50)
    for idx, row in shap_df.iterrows():
        bar = "#" * int(row['Mean_Abs_SHAP'] * 20)
        print(f"  {idx+1:>2}. {row['Feature']:<40} {row['Mean_Abs_SHAP']:.4f}  {bar}")

    os.makedirs(RESULT_DIR, exist_ok=True)
    out_csv = os.path.join(RESULT_DIR, "shap_feature_importance.csv")
    shap_df.to_csv(out_csv, index=False)
    print(f"\n  Saved SHAP ranking: {out_csv}")

    # 4. Global Beeswarm Summary Plot
    print("\n[Step 4] Generating SHAP Beeswarm Summary Plot...")
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    plt.figure(figsize=(10, 6.5), dpi=300)
    shap.summary_plot(shap_values.values, X_sample_df, feature_names=feature_names, show=False)
    plt.title("SHAP Beeswarm Summary Plot: Feature Impact on Default Risk (XGBoost)\nRed = High Feature Value, Blue = Low Feature Value",
              fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("SHAP Value (Impact on Log-Odds of Default)", fontsize=11, labelpad=8)
    plt.tight_layout()
    
    beeswarm_path = os.path.join(PLOT_DIR, "shap_summary_beeswarm.png")
    plt.savefig(beeswarm_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {beeswarm_path}")

    # 5. Feature Importance Bar Plot
    print("\n[Step 5] Generating SHAP Feature Importance Bar Plot...")
    plt.figure(figsize=(9, 5.5), dpi=300)
    y_pos = np.arange(len(shap_df))
    plt.barh(y_pos, shap_df["Mean_Abs_SHAP"][::-1], color="#2ca02c", edgecolor="#1b611b", height=0.6)
    plt.yticks(y_pos, shap_df["Feature"][::-1], fontsize=10)
    plt.xlabel("Mean |SHAP Value| (Average Impact on Model Output Magnitude)", fontsize=11, labelpad=8)
    plt.title("Global Feature Importance (TreeSHAP Ranking)", fontsize=13, fontweight="bold", pad=12)
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    
    bar_path = os.path.join(PLOT_DIR, "shap_feature_importance_bar.png")
    plt.savefig(bar_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {bar_path}")

    # 6. Non-linear Dependence Plot (Revolving Utilization)
    print("\n[Step 6] Generating SHAP Dependence Plot for Revolving Utilization...")
    plt.figure(figsize=(9, 6), dpi=300)
    util_idx = feature_names.index("RevolvingUtilizationOfUnsecuredLines")
    
    # Clip for clean visualization (un-standardized visualization)
    shap.dependence_plot(
        util_idx, shap_values.values, X_sample_df.values,
        feature_names=feature_names,
        show=False,
        interaction_index="NumberOfTimes90DaysLate"
    )
    plt.title("SHAP Dependence: Revolving Utilization vs. Default Risk\nColor: Number of Times 90 Days Late", fontsize=12, fontweight="bold", pad=12)
    plt.ylabel("SHAP Value for Revolving Utilization", fontsize=11, labelpad=8)
    plt.tight_layout()
    
    dep_path = os.path.join(PLOT_DIR, "shap_dependence_revolving_util.png")
    plt.savefig(dep_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {dep_path}")

    # 7. Local Explanations (Adverse Action Notice vs. Approval)
    print("\n[Step 7] Generating Individual Local Explanations (Waterfall Plots)...")
    probs = pipeline.predict_proba(X_sample_raw)[:, 1]
    
    # High risk applicant
    high_risk_idx = int(np.argmax(probs))
    low_risk_idx  = int(np.argmin(probs))

    # Waterfall 1: High Risk
    plt.figure(figsize=(9, 6), dpi=300)
    shap.plots.waterfall(shap_values[high_risk_idx], show=False)
    plt.title(f"Adverse Action Reason: High-Risk Applicant\nPredicted Default Probability = {probs[high_risk_idx]:.1%}",
              fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    waterfall_high = os.path.join(PLOT_DIR, "shap_waterfall_high_risk.png")
    plt.savefig(waterfall_high, bbox_inches="tight")
    plt.close()
    print(f"  Saved High-Risk Waterfall: {waterfall_high}")

    # Waterfall 2: Low Risk
    plt.figure(figsize=(9, 6), dpi=300)
    shap.plots.waterfall(shap_values[low_risk_idx], show=False)
    plt.title(f"Credit Approval Justification: Low-Risk Applicant\nPredicted Default Probability = {probs[low_risk_idx]:.1%}",
              fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    waterfall_low = os.path.join(PLOT_DIR, "shap_waterfall_low_risk.png")
    plt.savefig(waterfall_low, bbox_inches="tight")
    plt.close()
    print(f"  Saved Low-Risk Waterfall: {waterfall_low}")

    print("\n" + "=" * 65)
    print("PHASE 11 COMPLETE -- SHAP INTERPRETABILITY ANALYSIS FINISHED")
    print("=" * 65)


if __name__ == "__main__":
    run_shap_analysis()
