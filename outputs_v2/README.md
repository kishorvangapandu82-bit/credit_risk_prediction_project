# Research Reproduction Blueprint: Empirical Credit Default Prediction

**Project**: "An Empirical Evaluation of Machine Learning Models for Credit Default Prediction: Financial Stress Testing and SHAP-Based Interpretability"  
**Conference**: IEEE Conference on Machine Learning, Financial Engineering, and Data Science  
**Outputs Directory**: `outputs_v2/`

---

## 1. Environment & Dependencies

- **Python Version**: 3.11.9 (64-bit AMD64)
- **Primary Libraries**: `scikit-learn==1.9.1`, `xgboost==3.2.0`, `lightgbm==4.7.0`, `optuna==5.0.0`, `shap==0.51.0`, `numpy>=2.0.0`, `pandas>=2.2.0`, `matplotlib>=3.10.0`, `seaborn>=0.13.0`
- **Installation**:
  ```bash
  pip install -r outputs_v2/requirements.txt
  ```
- **Hardware Profile**: 20 Logical Processors (14 Physical Cores), 15.71 GB RAM.
- **Global Seed**: `42` (strictly applied to all cross-validation partitions and bootstrap resamples).

---

## 2. Directory Structure

```
credit-scoring-project/
├── data/
│   └── cs-training.csv                   # Raw Kaggle GMSC benchmark dataset (N=150,000)
├── models/                               # Serialized baseline models (joblib)
├── outputs_v2/
│   ├── figures/                          # Publication-grade figures (PDF + 300 dpi PNG)
│   │   ├── fig2_pipeline_workflow.[pdf|png]
│   │   ├── fig3_class_distribution.[pdf|png]
│   │   ├── fig4_confusion_matrices_v2.[pdf|png]
│   │   ├── fig5_roc_curves_v2.[pdf|png]
│   │   ├── fig6_stress_test_v2.[pdf|png]
│   │   ├── fig7_shap_summary_beeswarm_v2.[pdf|png]
│   │   ├── fig8_shap_bar_chart_v2.[pdf|png]
│   │   ├── precision_recall_curves_v2.[pdf|png]
│   │   ├── calibration_curves_top3.[pdf|png]
│   │   ├── feature_engineering_roc_v2.[pdf|png]
│   │   ├── shap_dependence_top4_v2.[pdf|png]
│   │   └── shap_waterfall_[high|low]_risk_v2.[pdf|png]
│   ├── tables/                           # Exact empirical outputs (CSV)
│   │   ├── repro_check.csv               # Baseline reproduction validation against paper
│   │   ├── thresholds.csv                # Training OOF optimal F1 and Youden thresholds
│   │   ├── table_iv_v2.csv               # Performance at cutoff 0.50 and optimized threshold
│   │   ├── extra_metrics.csv             # Gini, PR-AUC, KS statistic, Brier score
│   │   ├── bootstrap_ci.csv              # 1,000-sample bootstrap 95% CIs for holdout AUC
│   │   ├── significance_tests.csv        # DeLong and bootstrap pairwise p-values
│   │   ├── repeated_cv_results.csv       # Repeated Stratified 5-Fold x 5-Repeat CV (25 folds)
│   │   ├── search_spaces.csv             # Documented Optuna hyperparameter bounds
│   │   ├── best_params.csv               # Optuna best hyperparameters across all 7 models
│   │   ├── tuning_comparison.csv         # Grid Search vs Optuna TPE comparative audit
│   │   ├── scorecard_metrics.csv         # Traditional WOE Logistic Scorecard baseline
│   │   ├── stress_test_scenarios.csv     # 5 stress scenarios with recomputed DebtRatio
│   │   ├── stress_test_severity_sweep.csv# Severity sweep (-10% to -40% income cuts)
│   │   ├── engineered_features.csv       # Documentation of the 8 domain features
│   │   ├── feature_engineering_comparison.csv # 10-feature vs 18-feature holdout comparison
│   │   ├── shap_rankings.csv             # TreeSHAP feature attributions & Spearman correlation
│   │   ├── runtime_benchmark.csv         # Training time and inference latency per 1k rows
│   │   └── related_work.csv              # Literature benchmark table (replaces old Fig 1)
│   ├── logs/
│   │   ├── environment.txt               # Exact execution environment and package versions
│   │   └── preprocessing.md              # Full documentation of anti-leakage data processing
│   ├── requirements.txt                  # Python dependency specifications
│   ├── README.md                         # This execution guide
│   └── CHANGES.md                        # Master revision audit report of old vs new values
```

---

## 3. End-to-End Execution Sequence

To reproduce all results from scratch with deterministic random states:

```bash
# 0. Initial reproduction check
python -c "import pandas as pd; print(pd.read_csv('outputs_v2/tables/repro_check.csv'))"

# 1. Out-of-fold threshold tuning, extra metrics, and calibration / PR plots
python v2_pipeline/01_thresholds_and_extra_metrics.py

# 2. 1,000-sample bootstrap 95% CIs, DeLong tests, and repeated 5-fold CV (25 folds)
python v2_pipeline/02_statistical_rigor.py

# 3. Macroeconomic stress testing scenarios, severity sweep, and Figure 6
python v2_pipeline/04_stress_testing_v2.py

# 4. Domain-specific feature engineering (10 vs 18 features) and bootstrap Delta AUC
python v2_pipeline/05_feature_engineering_v2.py

# 5. TreeSHAP game-theoretic explainability, dependence, waterfall plots, and rank correlation
python v2_pipeline/06_shap_analysis_v2.py

# 6. Runtime computational complexity benchmark (training and inference latency)
python v2_pipeline/07_runtime_benchmark.py

# 7. Fair Bayesian optimization (Optuna TPE), Grid Search, and WOE Scorecard
python v2_pipeline/03_fair_tuning.py

# 8. Publication-grade figures (Fig 2, Fig 3, Fig 4, Fig 5) and related work table
python v2_pipeline/08_generate_figures.py
```
