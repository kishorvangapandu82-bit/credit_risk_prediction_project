# Comprehensive Empirical Audit & Baseline Reproduction Report

**File Target**: `AUDIT_AND_REPRODUCTION_RESULTS.md`  
**Workspace**: `c:/Users/SUMITRANAND SHARMA/OneDrive/Desktop/ML_PRO/credit-scoring-project`  
**Reference Document Audited**: `CONFERENCE_PAPER_DOCUMENTATION.md`  
**Dataset**: Kaggle *Give Me Some Credit* (GMSC), $N = 150,000$  
**Evaluation Date**: October 2026  

---

## Table of Contents
1. [Overview & Scope of Audit](#1-overview--scope-of-audit)
2. [Raw CSV Files for Each Table in CONFERENCE_PAPER_DOCUMENTATION.md](#2-raw-csv-files-for-each-table-in-conference_paper_documentationmd)
3. [Confirmation 1: Real Train/Test Split, Test-Set Size & Default Counts](#3-confirmation-1-real-traintest-split-test-set-size--default-counts)
4. [Confirmation 2: Exact Recall Calculation (k / n_defaults)](#4-confirmation-2-exact-recall-calculation-k--n_defaults)
5. [Confirmation 3: Spearman Rho Recomputed from shap_rankings.csv](#5-confirmation-3-spearman-rho-recomputed-from-shap_rankingscsv)
6. [Confirmation 4: Macroeconomic Stress Testing Recomputed on the 45,000 Test Set](#6-confirmation-4-macroeconomic-stress-testing-recomputed-on-the-45000-test-set)
7. [Search Spaces & The 48-Point Hyperparameter Grid](#7-search-spaces--the-48-point-hyperparameter-grid)
8. [Domain Feature Engineering Formulas & Rationale](#8-domain-feature-engineering-formulas--rationale)
9. [Verbatim environment.txt](#9-verbatim-environmenttxt)
10. [Verbatim CHANGES.md](#10-verbatim-changesmd)
11. [Exhaustive Inventory of Irreproducible Documented Numbers](#11-exhaustive-inventory-of-irreproducible-documented-numbers)

---

## 1. Overview & Scope of Audit

This audit document establishes the ground-truth empirical numbers for the credit default prediction project. It inspects:
- Every table presented in `CONFERENCE_PAPER_DOCUMENTATION.md`
- The raw CSV files in `outputs_v2/tables/` and `results/`
- The serialized scikit-learn/LightGBM/XGBoost models in `models/`
- The active preprocessing, pipeline, and split definitions in `src/` and `v2_pipeline/`

---

## 2. Raw CSV Files for Each Table in `CONFERENCE_PAPER_DOCUMENTATION.md`

Below is the verbatim raw CSV content corresponding to each table in `CONFERENCE_PAPER_DOCUMENTATION.md`.

### 2.1 Section 3 — Dataset Architecture & EDA
**Source**: Raw data file `data/cs-training.csv` via `src/inspect_data.py` and `src/split_verify.py`.
```text
Total Observations (N): 150000
Total Feature Dimensions: 10 predictive covariates + 1 target
Target Variable: SeriousDlqin2yrs
Negative Class Count (0): 139974 (93.3160%)
Positive Class Count (1): 10026 (6.6840%)
Class Imbalance Ratio: 13.9611 : 1
```

---

### 2.2 Section 6.2 — Empirical Tuning Results
**Source CSV 1**: `outputs_v2/tables/tuning_comparison.csv`
```csv
Model,Grid_CV_AUC,Grid_Test_AUC,Optuna_CV_AUC,Optuna_Test_AUC,Optuna_Delta_Test_AUC
XGBoost,0.8643,0.8661,0.8653,0.8669,0.0008
LightGBM,0.8649,0.867,0.8656,0.8671,0.0001
Random Forest,0.8626,0.8642,0.8636,0.8653,0.0011
```

**Source CSV 2**: `outputs_v2/tables/best_params.csv`
```csv
Model,Best_CV_AUC,Test_AUC,Best_Hyperparameters
XGBoost,0.8653,0.8669,"{'learning_rate': 0.03179049771163903, 'max_depth': 4, 'n_estimators': 350, 'subsample': 0.9482387762278873, 'colsample_bytree': 0.5666819122957921, 'gamma': 0.8038534866683684, 'reg_alpha': 0.004445403967360062, 'reg_lambda': 4.271903083597981, 'scale_pos_weight': 13.961527500712453}"
LightGBM,0.8656,0.8671,"{'learning_rate': 0.020487682947185094, 'num_leaves': 15, 'n_estimators': 350, 'min_child_samples': 43, 'reg_alpha': 8.131377408721187, 'reg_lambda': 0.5264255193978957, 'scale_pos_weight': 13.961527500712453}"
Random Forest,0.8636,0.8653,"{'n_estimators': 200, 'max_depth': 12, 'min_samples_split': 76, 'min_samples_leaf': 45, 'max_features': 'sqrt', 'class_weight': 'balanced'}"
Decision Tree,0.8502,0.8525,"{'max_depth': 7, 'min_samples_split': 98, 'min_samples_leaf': 50, 'criterion': 'entropy', 'class_weight': 'balanced'}"
Logistic Regression,0.8189,0.8226,"{'C': 0.06796788056943177, 'class_weight': 'balanced'}"
Linear SVM,0.8166,0.8209,"{'C': 0.00010109247362464303, 'class_weight': 'balanced'}"
MLP,0.8321,0.8344,"{'alpha': 0.00010374886356306705, 'learning_rate_init': 0.0020058755083587044, 'h_idx': 1, 'hidden_layer_sizes': (128, 64)}"
```

---

### 2.3 Section 8.1 — Holdout Performance Benchmark (Table IV Revised)
**Source CSV**: `outputs_v2/tables/table_iv_v2.csv`
```csv
Model,ROC_AUC,Cutoff_0.50_Recall,Cutoff_0.50_Precision,Cutoff_0.50_F1,Cutoff_0.50_Accuracy,Optimal_Threshold,OptThresh_Recall,OptThresh_Precision,OptThresh_F1,OptThresh_Accuracy,Youden_Threshold,Youden_Recall,Youden_Precision,Youden_F1,Youden_Accuracy
LightGBM,0.8667,0.7852,0.2126,0.3346,0.7912,0.76,0.5253,0.3856,0.4448,0.9123,0.4837,0.7949,0.204,0.3246,0.7789
XGBoost,0.8664,0.7773,0.2149,0.3367,0.7953,0.79,0.4801,0.4167,0.4462,0.9203,0.4801,0.7896,0.205,0.3255,0.7812
Random Forest,0.8639,0.7656,0.2179,0.3393,0.8007,0.73,0.515,0.3856,0.441,0.9127,0.4451,0.8122,0.1926,0.3113,0.7598
Decision Tree,0.8437,0.7354,0.2251,0.3447,0.8131,0.79,0.4817,0.394,0.4334,0.9158,0.5179,0.7354,0.2251,0.3447,0.8131
MLP,0.8376,0.7098,0.2205,0.3365,0.8129,0.23,0.9232,0.1034,0.186,0.4598,0.0598,0.9957,0.0691,0.1293,0.1035
Logistic Regression,0.8227,0.622,0.2702,0.3767,0.8624,0.66,0.4807,0.3786,0.4236,0.9126,0.4585,0.6772,0.2249,0.3377,0.8224
SVM,0.8205,0.1523,0.5902,0.2421,0.9363,0.12,0.4568,0.3967,0.4246,0.9172,0.0647,0.6536,0.2322,0.3427,0.8324
```

**Replication Validation CSV**: `outputs_v2/tables/repro_check.csv`
```csv
Model,Paper_AUC,New_AUC,Diff_AUC,Paper_Recall,New_Recall,Diff_Recall,Paper_Precision,New_Precision,Diff_Precision,Paper_F1,New_F1,Diff_F1,Paper_Accuracy,New_Accuracy,Diff_Accuracy,Average_Precision
LightGBM,0.8667,0.8667,0.0,0.7852,0.7852,0.0,0.2126,0.2126,-0.0,0.3346,0.3346,-0.0,0.7912,0.7912,0.0,0.4022
XGBoost,0.8664,0.8664,0.0,0.7773,0.7773,-0.0,0.2149,0.2149,-0.0,0.3367,0.3367,-0.0,0.7953,0.7953,-0.0,0.4029
Random Forest,0.8639,0.8639,0.0,0.7656,0.7656,0.0,0.2179,0.2179,0.0,0.3393,0.3393,0.0,0.8007,0.8007,-0.0,0.3933
Decision Tree,0.8437,0.8437,0.0,0.7354,0.7354,-0.0,0.2251,0.2251,0.0,0.3447,0.3447,0.0,0.8131,0.8131,0.0,0.3541
MLP,0.8376,0.8376,-0.0,0.7098,0.7098,-0.0,0.2205,0.2205,0.0,0.3365,0.3365,-0.0,0.8129,0.8129,-0.0,0.3655
Logistic Reg.,0.8227,0.8227,0.0,0.622,0.622,0.0,0.2702,0.2702,-0.0,0.3767,0.3767,0.0,0.8624,0.8624,0.0,0.3564
SVM,0.8205,0.8205,0.0,0.1523,0.1523,-0.0,0.5902,0.5902,0.0,0.2421,0.2421,-0.0,0.9363,0.9363,-0.0,0.3535
```

**Threshold Selection CSV**: `outputs_v2/tables/thresholds.csv`
```csv
Model,OOF_Best_F1,Optimal_F1_Threshold,Youden_J_Threshold
LightGBM,0.4405,0.76,0.4837
XGBoost,0.4382,0.79,0.4801
Random Forest,0.4311,0.73,0.4451
Decision Tree,0.4295,0.79,0.5179
MLP,0.4229,0.23,0.0598
Logistic Regression,0.4134,0.66,0.4585
SVM,0.4090,0.12,0.0647
```

---

### 2.4 Section 8.2 — Additional Credit Metrics (Gini, PR-AUC, KS, Brier)
**Source CSV**: `outputs_v2/tables/extra_metrics.csv`
```csv
Model,ROC_AUC,Gini_Coefficient,PR_AUC_Average_Precision,KS_Statistic,Brier_Score
LightGBM,0.8667,0.7335,0.4022,0.5808,0.1423
XGBoost,0.8664,0.7329,0.4029,0.5756,0.141
Random Forest,0.8639,0.7279,0.3933,0.5721,0.1364
Decision Tree,0.8437,0.6874,0.3541,0.5541,0.1483
MLP,0.8376,0.6752,0.3655,0.5318,0.1472
Logistic Regression,0.8227,0.6454,0.3564,0.5119,0.1548
SVM,0.8205,0.641,0.3535,0.5029,0.0528
```

**Scorecard Baseline CSV**: `outputs_v2/tables/scorecard_metrics.csv`
```csv
Model,ROC_AUC,Gini,Recall,Precision,F1,Accuracy,Binning_Strategy,Weighting
WOE Logistic Scorecard,0.828,0.6561,0.758,0.1791,0.2897,0.7516,8 Quantile Bins with Missing Category,balanced
```

---

### 2.5 Section 8.3 — Statistical Hypothesis Testing: DeLong & Paired Bootstrap Tests
**Source CSV**: `outputs_v2/tables/significance_tests.csv`
```csv
Model_A,Model_B,AUC_Diff (A - B),DeLong_SE,DeLong_Z,DeLong_p_value,Bootstrap_Diff_95_CI,Bootstrap_p_value,Significant_at_0.05
LightGBM,XGBoost,0.00029,0.00048,0.616,0.53791,"[-0.00056, 0.00120]",0.534,No
LightGBM,Random Forest,0.00278,0.00058,4.7661,0.0,"[0.00167, 0.00388]",0.001,Yes
XGBoost,Random Forest,0.00248,0.00064,3.8667,0.00011,"[0.00127, 0.00363]",0.001,Yes
```

**1,000-Iteration Bootstrap CI CSV**: `outputs_v2/tables/bootstrap_ci.csv`
```csv
Model,Holdout_AUC,Bootstrap_Mean,Bootstrap_SE,CI_95_Lower,CI_95_Upper,CI_95_Formatted
LightGBM,0.8667,0.8666,0.0035,0.86,0.8735,"[0.8600, 0.8735]"
XGBoost,0.8664,0.8663,0.0035,0.8596,0.873,"[0.8596, 0.8730]"
Random Forest,0.8639,0.8638,0.0035,0.8571,0.8704,"[0.8571, 0.8704]"
Decision Tree,0.8437,0.8436,0.0039,0.8361,0.8509,"[0.8361, 0.8509]"
MLP,0.8376,0.8375,0.0041,0.8289,0.8453,"[0.8289, 0.8453]"
Logistic Regression,0.8227,0.8226,0.0044,0.8141,0.8312,"[0.8141, 0.8312]"
SVM,0.8205,0.8204,0.0044,0.8119,0.8291,"[0.8119, 0.8291]"
```

**Repeated CV CSV**: `outputs_v2/tables/repeated_cv_results.csv`
```csv
Model,Mean_CV_AUC,Std_CV_AUC,Min_CV_AUC,Max_CV_AUC,Folds,Repeats
LightGBM,0.8632,0.0053,0.8521,0.8719,5,5
XGBoost,0.8628,0.0052,0.8519,0.8712,5,5
Random Forest,0.8617,0.0053,0.8508,0.8701,5,5
```

---

### 2.6 Section 9 — Feature Engineering (10 vs. 18 Features)
**Source CSV**: `outputs_v2/tables/feature_engineering_comparison.csv`
```csv
Model,Features_10_AUC,Features_18_AUC,Delta_AUC,Delta_AUC_95_CI,Bootstrap_p_value,Features_10_F1,Features_18_F1,Features_10_Precision,Features_18_Precision,Features_10_Recall,Features_18_Recall,Features_10_Accuracy,Features_18_Accuracy
Random Forest,0.8639,0.8637,-0.0003,"[-0.0009, 0.0004]",0.366,0.3393,0.3485,0.2179,0.2267,0.7656,0.753,0.8007,0.8118
XGBoost,0.8664,0.8657,-0.0008,"[-0.0018, 0.0001]",0.106,0.3367,0.3403,0.2149,0.2182,0.7773,0.7729,0.7953,0.7997
LightGBM,0.8667,0.8649,-0.0019,"[-0.0030, -0.0008]",0.001,0.3346,0.3432,0.2126,0.2212,0.7852,0.765,0.7912,0.8043
```

---

### 2.7 Section 10.2 — Macroeconomic Stress Testing Scenarios
**Source CSV 1**: `outputs_v2/tables/stress_test_scenarios.csv`
```csv
Scenario,Model,Baseline_AUC,Stressed_AUC,Retained_AUC_Pct,Baseline_Mean_Prob,Stressed_Mean_Prob,Mean_Prob_Increase_Pct,Optimal_Threshold,Baseline_Flagged_Rate,Stressed_Flagged_Rate,Flagged_Rate_Change_Pct
(a) Income -20% (Mechanically Recomputed DebtRatio),LightGBM,0.8667,0.866,99.914,0.3195,0.3315,3.76,0.76,0.091,0.0935,2.73
(a) Income -20% (Mechanically Recomputed DebtRatio),XGBoost,0.8664,0.866,99.95,0.3219,0.3331,3.51,0.79,0.077,0.081,5.14
(a) Income -20% (Mechanically Recomputed DebtRatio),Random Forest,0.8639,0.8634,99.936,0.3181,0.3273,2.88,0.73,0.0893,0.0916,2.64
(a) Income -20% (Mechanically Recomputed DebtRatio),Decision Tree,0.8437,0.8438,100.006,0.3311,0.331,-0.01,0.79,0.0817,0.0821,0.41
(a) Income -20% (Mechanically Recomputed DebtRatio),MLP,0.8376,0.8372,99.959,0.3378,0.3491,3.35,0.23,0.5968,0.6292,5.43
(a) Income -20% (Mechanically Recomputed DebtRatio),Logistic Regression,0.8227,0.8221,99.928,0.3751,0.3784,0.88,0.66,0.0849,0.0856,0.86
(a) Income -20% (Mechanically Recomputed DebtRatio),SVM,0.8205,0.8198,99.913,0.0665,0.0673,1.22,0.12,0.077,0.0779,1.21
(a*) Income -20% (Quirk-Aware: non-null only),LightGBM,0.8667,0.8661,99.934,0.3195,0.3315,3.77,0.76,0.091,0.0936,2.83
(a*) Income -20% (Quirk-Aware: non-null only),XGBoost,0.8664,0.866,99.956,0.3219,0.3332,3.52,0.79,0.077,0.081,5.14
(a*) Income -20% (Quirk-Aware: non-null only),Random Forest,0.8639,0.8635,99.946,0.3181,0.3273,2.89,0.73,0.0893,0.092,3.09
(a*) Income -20% (Quirk-Aware: non-null only),Decision Tree,0.8437,0.8438,100.013,0.3311,0.3311,0.02,0.79,0.0817,0.0821,0.46
(a*) Income -20% (Quirk-Aware: non-null only),MLP,0.8376,0.8374,99.982,0.3378,0.3484,3.15,0.23,0.5968,0.6268,5.02
(a*) Income -20% (Quirk-Aware: non-null only),Logistic Regression,0.8227,0.8224,99.958,0.3751,0.3791,1.07,0.66,0.0849,0.0858,1.1
(a*) Income -20% (Quirk-Aware: non-null only),SVM,0.8205,0.8201,99.953,0.0665,0.0674,1.47,0.12,0.077,0.0782,1.59
(b) DebtRatio +25% Only (Constant Income),LightGBM,0.8667,0.8662,99.934,0.3195,0.3267,2.26,0.76,0.091,0.0922,1.27
(b) DebtRatio +25% Only (Constant Income),XGBoost,0.8664,0.8662,99.969,0.3219,0.328,1.92,0.79,0.077,0.0794,3.06
(b) DebtRatio +25% Only (Constant Income),Random Forest,0.8639,0.8635,99.946,0.3181,0.3236,1.73,0.73,0.0893,0.0902,1.07
(b) DebtRatio +25% Only (Constant Income),Decision Tree,0.8437,0.8437,99.994,0.3311,0.3309,-0.03,0.79,0.0817,0.0817,-0.05
(b) DebtRatio +25% Only (Constant Income),MLP,0.8376,0.8373,99.966,0.3378,0.3384,0.2,0.23,0.5968,0.5991,0.38
(b) DebtRatio +25% Only (Constant Income),Logistic Regression,0.8227,0.8225,99.974,0.3751,0.3743,-0.21,0.66,0.0849,0.0847,-0.24
(b) DebtRatio +25% Only (Constant Income),SVM,0.8205,0.8202,99.961,0.0665,0.0663,-0.27,0.12,0.077,0.0766,-0.43
(c) Original Combined (-20% Inc & +25% Debt),LightGBM,0.8667,0.866,99.914,0.3195,0.3315,3.76,0.76,0.091,0.0935,2.73
(c) Original Combined (-20% Inc & +25% Debt),XGBoost,0.8664,0.866,99.95,0.3219,0.3331,3.51,0.79,0.077,0.081,5.14
(c) Original Combined (-20% Inc & +25% Debt),Random Forest,0.8639,0.8634,99.936,0.3181,0.3273,2.88,0.73,0.0893,0.0916,2.64
(c) Original Combined (-20% Inc & +25% Debt),Decision Tree,0.8437,0.8438,100.006,0.3311,0.331,-0.01,0.79,0.0817,0.0821,0.41
(c) Original Combined (-20% Inc & +25% Debt),MLP,0.8376,0.8372,99.959,0.3378,0.3491,3.35,0.23,0.5968,0.6292,5.43
(c) Original Combined (-20% Inc & +25% Debt),Logistic Regression,0.8227,0.8221,99.928,0.3751,0.3784,0.88,0.66,0.0849,0.0856,0.86
(c) Original Combined (-20% Inc & +25% Debt),SVM,0.8205,0.8198,99.913,0.0665,0.0673,1.22,0.12,0.077,0.0779,1.21
(e) Targeted Shock (-20% Inc to Bottom 30%),LightGBM,0.8667,0.8667,99.994,0.3195,0.3208,0.4,0.76,0.091,0.0915,0.46
(e) Targeted Shock (-20% Inc to Bottom 30%),XGBoost,0.8664,0.8664,99.991,0.3219,0.323,0.35,0.79,0.077,0.0775,0.61
(e) Targeted Shock (-20% Inc to Bottom 30%),Random Forest,0.8639,0.8638,99.988,0.3181,0.3188,0.19,0.73,0.0893,0.0891,-0.2
(e) Targeted Shock (-20% Inc to Bottom 30%),Decision Tree,0.8437,0.8438,100.007,0.3311,0.331,-0.02,0.79,0.0817,0.0817,-0.05
(e) Targeted Shock (-20% Inc to Bottom 30%),MLP,0.8376,0.8377,100.008,0.3378,0.3394,0.48,0.23,0.5968,0.6001,0.55
(e) Targeted Shock (-20% Inc to Bottom 30%),Logistic Regression,0.8227,0.8228,100.014,0.3751,0.3755,0.1,0.66,0.0849,0.085,0.1
(e) Targeted Shock (-20% Inc to Bottom 30%),SVM,0.8205,0.8206,100.011,0.0665,0.0666,0.18,0.12,0.077,0.0772,0.35
```

**Source CSV 2 (Baseline Table VIII from `src/stress_testing.py`)**: `results/stress_test_comparison.csv`
```csv
Model,Baseline_AUC,Stressed_AUC,AUC_Drop,AUC_Retention_%,Baseline_Mean_Prob,Stressed_Mean_Prob,Prob_Increase_%,Baseline_Pred_Default_Rate,Stressed_Pred_Default_Rate,Default_Rate_Surge_%
Logistic Regression,0.8227,0.8221,0.0006,99.93,0.3751,0.3784,0.88,0.1539,0.1562,1.47
Decision Tree,0.8437,0.8438,-0.0,100.01,0.3311,0.331,-0.01,0.2183,0.2182,-0.06
MLP Neural Network,0.8376,0.8372,0.0003,99.96,0.3378,0.3491,3.35,0.2152,0.2246,4.38
XGBoost,0.8664,0.866,0.0004,99.95,0.32190001010894775,0.33309999108314514,3.509999990463257,0.2418,0.2566,6.14
LightGBM,0.8667,0.866,0.0007,99.91,0.3195,0.3315,3.76,0.2469,0.262,6.09
Support Vector Machine,0.8205,0.8198,0.0007,99.91,0.0665,0.0673,1.22,0.0172,0.0173,0.13
Random Forest,0.8639,0.8634,0.0006,99.94,0.3181,0.3273,2.88,0.2348,0.2486,5.86
```

---

### 2.8 Section 11.1 — Global TreeSHAP Feature Attributions
**Source CSV**: `outputs_v2/tables/shap_rankings.csv`
```csv
Feature,LightGBM_Mean_Abs_SHAP,LightGBM_Rank,XGBoost_Mean_Abs_SHAP,XGBoost_Rank,Spearman_Rho,Spearman_p_value
RevolvingUtilizationOfUnsecuredLines,0.7171500983895928,1,0.8255769,1,0.9394,5.5e-05
NumberOfTime30-59DaysPastDueNotWorse,0.30322699312963325,2,0.3588834,2,0.9394,5.5e-05
age,0.2466799272091989,3,0.23619632,4,0.9394,5.5e-05
NumberOfTimes90DaysLate,0.22880960660903144,4,0.32396296,3,0.9394,5.5e-05
NumberOfOpenCreditLinesAndLoans,0.1595007385088728,5,0.16488059,6,0.9394,5.5e-05
DebtRatio,0.12126715878736234,6,0.12940483,7,0.9394,5.5e-05
NumberOfTime60-89DaysPastDueNotWorse,0.1209169262310583,7,0.18035388,5,0.9394,5.5e-05
NumberRealEstateLoansOrLines,0.09683828092118818,8,0.10100498,9,0.9394,5.5e-05
MonthlyIncome,0.09209174891560613,9,0.11257166,8,0.9394,5.5e-05
NumberOfDependents,0.010240938050863522,10,0.018644402,10,0.9394,5.5e-05
```

---

### 2.9 Section 15 — Computational Latency & System Footprint
**Source CSV**: `outputs_v2/tables/runtime_benchmark.csv`
```csv
Model,Training_Time_Sec,Inference_Time_Full_Test_Sec,Inference_Latency_ms_per_1k_Rows,Hardware_CPU_Cores,System_RAM_GB
Logistic Regression,0.11,0.013,0.28,20,15.71
Decision Tree,0.23,0.012,0.26,20,15.71
SVM (Calibrated),0.26,0.018,0.39,20,15.71
MLP Neural Network,3.08,0.026,0.57,20,15.71
Random Forest,1.46,0.09,2.0,20,15.71
XGBoost,0.39,0.026,0.57,20,15.71
LightGBM,0.48,0.049,1.08,20,15.71
```

---

## 3. Confirmation 1: Real Train/Test Split, Test-Set Size & Default Counts

1. **The Real Split Used in Code Was Stratified 70:30 (Not 80/20)**:
   - Config file `src/config.py`: `TEST_SIZE = 0.30`
   - Split verification script `src/split_verify.py`: executes `test_size=0.30`
   - Metadata document `outputs_v2/logs/environment.txt`: specifies `Stratified 70:30 (105,000 Train / 45,000 Test)`
   - Preprocessing report `outputs_v2/logs/preprocessing.md`: confirms `Stratified 70:30 split`
   - The serialized models in `models/` yield exact documented AUCs (`0.8667`, `0.8664`, `0.8639`, etc.) **only** when evaluated on the 45,000 test set.

2. **Exact Numerical Partitioning**:
   - **Full Dataset ($N = 150,000$)**:
     - Non-defaults ($y=0$): **139,974** ($93.3160\%$)
     - Defaults ($y=1$): **10,026** ($6.6840\%$)
   - **Real Training Split (70%, $N_{\text{train}} = 105,000$)**:
     - Non-defaults ($y=0$): **97,982** ($93.3162\%$)
     - Defaults ($y=1$): **7,018** ($6.6838\%$)
   - **Real Holdout Test Split (30%, $N_{\text{test}} = 45,000$)**:
     - Non-defaults ($y=0$): **41,992** ($93.3156\%$)
     - Defaults ($y=1$): **3,008** ($6.6844\%$)
   - **Hypothetical 80/20 Split (Erroneously Documented in Paper Text)**:
     - Training partition ($80\%$): $N = 120,000$, Non-defaults = $111,979$, Defaults = $8,021$
     - Test partition ($20\%$): $N = 30,000$, Non-defaults = $27,995$, Defaults = $2,005$

---

## 4. Confirmation 2: Exact Recall Calculation ($k / n_{\text{defaults}}$)

On the real holdout test set ($N = 45,000$), the number of default observations is **$n_{\text{defaults}} = 3,008$**.

Every single recall value in `table_iv_v2.csv` and `repro_check.csv` equals an exact integer count of caught defaults $k$ divided by $3,008$:

### A. Threshold $\tau = 0.50$:
- **LightGBM**: $k = 2,362 \implies 2362 / 3008 = 0.785239... \implies \mathbf{0.7852}$
- **XGBoost**: $k = 2,338 \implies 2338 / 3008 = 0.777261... \implies \mathbf{0.7773}$
- **Random Forest**: $k = 2,303 \implies 2303 / 3008 = 0.765625... \implies \mathbf{0.7656}$
- **Decision Tree**: $k = 2,212 \implies 2212 / 3008 = 0.735372... \implies \mathbf{0.7354}$
- **MLP**: $k = 2,135 \implies 2135 / 3008 = 0.709774... \implies \mathbf{0.7098}$
- **Logistic Regression**: $k = 1,871 \implies 1871 / 3008 = 0.622008... \implies \mathbf{0.6220}$
- **Linear SVM**: $k = 458 \implies 458 / 3008 = 0.152261... \implies \mathbf{0.1523}$

### B. Optimal OOF Threshold $\tau^*$:
- **LightGBM** ($\tau^* = 0.76$): $k = 1,580 \implies 1580 / 3008 = 0.525266... \implies \mathbf{0.5253}$
- **XGBoost** ($\tau^* = 0.79$): $k = 1,444 \implies 1444 / 3008 = 0.480053... \implies \mathbf{0.4801}$
- **Random Forest** ($\tau^* = 0.73$): $k = 1,549 \implies 1549 / 3008 = 0.514960... \implies \mathbf{0.5150}$
- **Decision Tree** ($\tau^* = 0.79$): $k = 1,449 \implies 1449 / 3008 = 0.481715... \implies \mathbf{0.4817}$
- **MLP** ($\tau^* = 0.23$): $k = 2,777 \implies 2777 / 3008 = 0.923205... \implies \mathbf{0.9232}$
- **Logistic Regression** ($\tau^* = 0.66$): $k = 1,446 \implies 1446 / 3008 = 0.480718... \implies \mathbf{0.4807}$
- **Linear SVM** ($\tau^* = 0.12$): $k = 1,374 \implies 1374 / 3008 = 0.456782... \implies \mathbf{0.4568}$

### C. Evaluation of Erroneous Documentation Numbers:
The values printed in `CONFERENCE_PAPER_DOCUMENTATION.md` Table 8.1 (`Recall = 0.7712, 0.7707, 0.7588, 0.6874, 0.7107`) do **not** equal $k / 3,008$ or $k / 2,005$ for any integer $k$. They are ungrounded artifacts.

---

## 5. Confirmation 3: Spearman Rho Recomputed from `shap_rankings.csv`

Recomputing Spearman rank correlation on the 10 feature importance scores in `outputs_v2/tables/shap_rankings.csv`:

| Feature Name | LightGBM Rank ($R_1$) | XGBoost Rank ($R_2$) | Difference $d_i$ | $d_i^2$ |
| :--- | :---: | :---: | :---: | :---: |
| `RevolvingUtilizationOfUnsecuredLines` | 1 | 1 | 0 | 0 |
| `NumberOfTime30-59DaysPastDueNotWorse` | 2 | 2 | 0 | 0 |
| `age` | 3 | 4 | -1 | 1 |
| `NumberOfTimes90DaysLate` | 4 | 3 | +1 | 1 |
| `NumberOfOpenCreditLinesAndLoans` | 5 | 6 | -1 | 1 |
| `DebtRatio` | 6 | 7 | -1 | 1 |
| `NumberOfTime60-89DaysPastDueNotWorse` | 7 | 5 | +2 | 4 |
| `NumberRealEstateLoansOrLines` | 8 | 9 | -1 | 1 |
| `MonthlyIncome` | 9 | 8 | +1 | 1 |
| `NumberOfDependents` | 10 | 10 | 0 | 0 |
| **Sum** | | | | $\sum d_i^2 = 10$ |

$$\rho = 1 - \frac{6 \times 10}{10(10^2 - 1)} = 1 - \frac{60}{990} = 1 - \frac{2}{33} = \frac{31}{33} \approx 0.9393939...$$

- **Recomputed Spearman $\rho$**: **$0.9394$**
- **Two-tailed $p$-value**: **$5.48405 \times 10^{-5}$** ($5.5 \times 10^{-5}$)

---

## 6. Confirmation 4: Macroeconomic Stress Testing Recomputed on the 45,000 Test Set

Evaluating the simulated macroeconomic shock ($-20\%$ static MonthlyIncome cut with analytical $+25\%$ DebtRatio expansion under constant nominal debt obligations) on the identical 45,000 test set:

| Model Architecture | Baseline Mean Prob | Stressed Mean Prob | Relative Prob Shift (%) | Baseline Flagged Share ($\tau=0.50$) | Stressed Flagged Share ($\tau=0.50$) | Relative Default Surge (%) | Baseline Test AUC | Stressed Test AUC | Retained AUC (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM** | 0.3195 | 0.3315 | **+3.76%** | 24.69% (11,111 / 45k) | 26.20% (11,790 / 45k) | **+6.09%** | **0.8667** | **0.8660** | **99.914%** |
| **XGBoost** | 0.3219 | 0.3331 | **+3.51%** | 24.18% (10,881 / 45k) | 25.66% (11,547 / 45k) | **+6.14%** | **0.8664** | **0.8660** | **99.950%** |
| **Random Forest** | 0.3181 | 0.3273 | +2.88% | 23.48% (10,567 / 45k) | 24.86% (11,186 / 45k) | +5.86% | 0.8639 | 0.8634 | 99.936% |
| **Decision Tree** | 0.3311 | 0.3310 | -0.01% | 21.83% (9,825 / 45k) | 21.82% (9,819 / 45k) | -0.06% | 0.8437 | 0.8438 | 100.006% |
| **MLP** | 0.3378 | 0.3491 | +3.35% | 21.52% (9,684 / 45k) | 22.46% (10,107 / 45k) | +4.38% | 0.8376 | 0.8372 | 99.959% |
| **Logistic Regression** | 0.3751 | 0.3784 | +0.88% | 15.39% (6,926 / 45k) | 15.62% (7,028 / 45k) | +1.47% | 0.8227 | 0.8221 | 99.928% |
| **SVM** | 0.0665 | 0.0673 | +1.22% | 1.72% (774 / 45k) | 1.73% (779 / 45k) | +0.13% | 0.8205 | 0.8198 | 99.913% |

### Resolution of the Misdocumented Figures:
- The values $+6.09\%$ and $+6.14\%$ were the **relative percentage increases in flagged defaults** at the 0.50 threshold:
  $$\frac{0.2620 - 0.2469}{0.2469} = +6.09\%, \quad \frac{0.2566 - 0.2418}{0.2418} = +6.14\%$$
- The paper documentation erroneously converted them into absolute differences ($17.52\% \to 23.61\%$), which were completely invented.
- The true retained AUCs are **0.8660** (LightGBM) and **0.8660** (XGBoost), retaining $>99.9\%$ of baseline ranking capacity, not the fabricated $0.8530$ and $0.8524$.

---

## 7. Search Spaces & The 48-Point Hyperparameter Grid

### 7.1 Raw CSV: `outputs_v2/tables/search_spaces.csv`
```csv
Model,Tuning_Protocol,Hyperparameter,Type,Range_or_Values
XGBoost,"Optuna TPE (50 trials, 5-fold CV AUC)",learning_rate,float (log),"[0.01, 0.20]"
XGBoost,Optuna TPE,max_depth,int,"[3, 8]"
XGBoost,Optuna TPE,n_estimators,int,"[100, 400]"
XGBoost,Optuna TPE,subsample,float,"[0.60, 1.00]"
XGBoost,Optuna TPE,colsample_bytree,float,"[0.50, 1.00]"
XGBoost,Optuna TPE,gamma,float,"[0.00, 1.00]"
XGBoost,Optuna TPE,reg_alpha,float (log),"[1e-3, 10.0]"
XGBoost,Optuna TPE,reg_lambda,float (log),"[1e-3, 10.0]"
XGBoost,Optuna TPE,scale_pos_weight,constant,13.96
LightGBM,Optuna TPE,learning_rate,float (log),"[0.01, 0.20]"
LightGBM,Optuna TPE,num_leaves,int,"[15, 127]"
LightGBM,Optuna TPE,n_estimators,int,"[100, 400]"
LightGBM,Optuna TPE,min_child_samples,int,"[10, 100]"
LightGBM,Optuna TPE,reg_alpha,float (log),"[1e-3, 10.0]"
LightGBM,Optuna TPE,reg_lambda,float (log),"[1e-3, 10.0]"
LightGBM,Optuna TPE,scale_pos_weight,constant,13.96
Random Forest,Optuna TPE,n_estimators,int,"[100, 300]"
Random Forest,Optuna TPE,max_depth,int,"[6, 20]"
Random Forest,Optuna TPE,min_samples_split,int,"[10, 100]"
Random Forest,Optuna TPE,min_samples_leaf,int,"[5, 50]"
Random Forest,Optuna TPE,max_features,categorical,"{sqrt, log2}"
Random Forest,Optuna TPE,class_weight,constant,balanced
Decision Tree,Optuna TPE,max_depth,int,"[3, 12]"
Decision Tree,Optuna TPE,min_samples_split,int,"[10, 100]"
Decision Tree,Optuna TPE,min_samples_leaf,int,"[5, 50]"
Decision Tree,Optuna TPE,criterion,categorical,"{gini, entropy}"
Decision Tree,Optuna TPE,class_weight,constant,balanced
Logistic Regression,Optuna TPE,C,float (log),"[1e-4, 10.0]"
Logistic Regression,Optuna TPE,class_weight,constant,balanced
Linear SVM,Optuna TPE,C,float (log),"[1e-4, 10.0]"
Linear SVM,Optuna TPE,class_weight,constant,balanced
MLP,Optuna TPE,alpha,float (log),"[1e-4, 1e-1]"
MLP,Optuna TPE,learning_rate_init,float (log),"[1e-4, 1e-2]"
MLP,Optuna TPE,hidden_layer_sizes,categorical,"{(64, 32), (128, 64), (64,), (100,)}"
```

### 7.2 The 48-Point Hyperparameter Grids Evaluated in `v2_pipeline/03_fair_tuning.py`
1. **XGBoost Grid (48 Combinations = $4 \times 4 \times 3$)**:
   - `clf__learning_rate`: `[0.01, 0.05, 0.1, 0.2]`
   - `clf__max_depth`: `[3, 4, 5, 6]`
   - `clf__n_estimators`: `[100, 200, 300]`
   - Fixed: `scale_pos_weight = 13.96`, `random_state = 42`
2. **LightGBM Grid (48 Combinations = $4 \times 4 \times 3$)**:
   - `clf__learning_rate`: `[0.01, 0.05, 0.1, 0.2]`
   - `clf__num_leaves`: `[15, 31, 63, 127]`
   - `clf__n_estimators`: `[100, 200, 300]`
   - Fixed: `scale_pos_weight = 13.96`, `random_state = 42`
3. **Random Forest Grid (48 Combinations = $3 \times 4 \times 4$)**:
   - `clf__n_estimators`: `[100, 200, 300]`
   - `clf__max_depth`: `[6, 10, 14, 18]`
   - `clf__min_samples_split`: `[10, 30, 50, 100]`
   - Fixed: `class_weight = "balanced"`, `random_state = 42`

---

## 8. Domain Feature Engineering Formulas & Rationale

### Raw CSV: `outputs_v2/tables/engineered_features.csv`
```csv
Feature_Name,Mathematical_Formula,Economic_Rationale,Leakage_Protocol
TotalPastDueCount,NumberOfTime30-59Days + NumberOfTime60-89Days + NumberOfTimes90DaysLate,Aggregates overall delinquency incidence across all temporal severity buckets to capture cumulative distress frequency.,Strictly row-wise aggregation; no cross-observation statistics utilized.
DelinquencySeverityIndex,1.0*(30-59Days) + 2.0*(60-89Days) + 3.0*(90+DaysLate),"Weights severe default signals linearly higher (90+ days weighted 3x vs 30-day delays), reflecting ordinal risk escalation.",Strictly row-wise linear combination; zero cross-observation leakage.
IncomePerDependent,MonthlyIncome_imputed / (NumberOfDependents_imputed + 1.0),Normalizes gross household purchasing power per dependent member to evaluate per-capita subsistence margin.,Imputation median and mode computed strictly on the training partition.
EstimatedMonthlyDebt,"MonthlyIncome_imputed * clip(DebtRatio, upper=10.0)",Converts dimensionless debt ratio into estimated dollar-denominated monthly debt service obligations.,Income median fit on training data; ratio clipping bound is constant.
DisposableIncome,MonthlyIncome_imputed - EstimatedMonthlyDebt,Estimates free residual cash flow buffer available to absorb emergency financial shocks.,Derived from training-imputed income and debt estimates; zero test leakage.
UnsecuredLinesRatio,(NumberOfOpenCreditLines - NumberRealEstateLoans) / (NumberOfOpenCreditLines + 1.0),Isolates uncollateralized revolving credit cards from secured real estate mortgages to measure debt vulnerability.,Strictly row-wise operation on applicant loan counts.
MissingIncomeFlag,Indicator(MonthlyIncome is NaN),"Flags unverified or undocumented income applications, which strongly correlate with non-traditional or high-risk employment.",Identified prior to imputation on raw applicant entries.
OverUtilizationFlag,Indicator(RevolvingUtilizationOfUnsecuredLines > 1.0),"Identifies applicants currently exceeding their total approved credit limit, a primary distress indicator.",Row-wise threshold comparison; completely leakage-free.
```

---

## 9. Verbatim `environment.txt`

The complete raw text of `outputs_v2/logs/environment.txt`:

```text
=============================================================================
COMPUTATIONAL ENVIRONMENT & REPRODUCIBILITY METADATA
=============================================================================
Operating System: Windows 10 (Build 26200, SP0, 64-bit AMD64)
Python Version  : 3.11.9 (tags/v3.11.9:de54cf5, Apr 2 2024, 10:12:12) [MSC v.1938 64 bit (AMD64)]
CPU Architecture: 20 Logical Processors, 14 Physical Cores
System Memory   : 15.71 GB Physical RAM

Core Scientific Library Versions:
  - scikit-learn : 1.9.1
  - xgboost      : 3.2.0
  - lightgbm     : 4.7.0
  - optuna       : 5.0.0
  - shap         : 0.51.0
  - numpy        : 2.2.6
  - pandas       : 2.2.3
  - scipy        : 1.15.2
  - matplotlib   : 3.10.1
  - seaborn      : 0.13.2
  - joblib       : 1.4.2

Random Seeds & Partitions:
  - Global Random Seed  : 42
  - Train / Holdout Split: Stratified 70:30 (105,000 Train / 45,000 Test)
  - Cross-Validation     : Stratified K-Fold (5 folds for Optuna, 10 folds for baseline CV)
  - Bootstrap Iterations : 1,000 paired resamples
=============================================================================
```

---

## 10. Verbatim `CHANGES.md`

The complete raw markdown text of `outputs_v2/CHANGES.md`:

```markdown
# Comprehensive Revision Audit Report: Methodological Corrections & Empirical Re-Evaluation

**Document Target**: IEEE Conference Paper Revision  
**Paper Title**: *An Empirical Evaluation of Machine Learning Models for Credit Default Prediction: Financial Stress Testing and SHAP-Based Interpretability*  
**Dataset**: Kaggle *Give Me Some Credit* (GMSC), $N = 150,000$ (120,000 train / 30,000 test, stratified 80/20, seed 42)  
**Revision Date**: October 2026  
**Pipeline Location**: `v2_pipeline/`  
**Output Directory**: `outputs_v2/` (`tables/`, `figures/`, `logs/`)  

---

## Executive Summary

This audit report details the reproduction, rigorous methodological correction, and empirical re-evaluation of the 7 machine learning models, stress testing protocols, and explainability analyses presented in the conference paper. 

### Key Audit Findings:
1. **Exact Baseline Reproduction**: At the fixed seed 42, all 7 baseline models reproduce the original Table IV metrics with **zero discrepancy** ($0.0000$ error across Accuracy, Precision, Recall, F1, and ROC-AUC).
2. **Threshold Optimization & The SVM "Artifact" Resolved**: The paper reported an SVM recall of 0.1523 at the default 0.50 probability cutoff, concluding SVM is ineffective. We prove this was a **pure threshold artifact** caused by Platt scaling on an imbalanced dataset (6.68% default rate). When optimizing the decision threshold on training Out-Of-Fold (OOF) cross-validation predictions ($\tau^* = 0.12$), SVM achieves a Recall of **0.4568** and F1 of **0.4246**, rivaling gradient boosted ensembles.
3. **Statistical Significance (DeLong & Paired Bootstrap)**: The paper claimed LightGBM ($0.8667$) was definitively superior to XGBoost ($0.8664$). Formal DeLong hypothesis testing reveals $Z = 0.6160, p = 0.53791$ (paired bootstrap $p = 0.52800$). **LightGBM and XGBoost are statistically indistinguishable.** Both are, however, statistically superior to Random Forest ($p < 0.001$).
4. **Resolution of Double-Counting Stress Test Logic**: The original code applied a $-20\%$ income shock and a separate $+20\%$ debt shock, compounding DebtRatio to $+56.25\%$. Economically, under sticky nominal debt ($D_t = D_0$), a $20\%$ income drop mechanically increases DebtRatio by exactly $+25\%$ ($1 / 0.80 = 1.25$). Correcting this logic shows that mean default probabilities increase by $+3.76\%$ (LightGBM) and $+3.51\%$ (XGBoost), with retained AUCs of $0.8530$ and $0.8524$. The paper's Table VIII numbers ($+6.09\%$ and $+6.14\%$) were mislabeled: they represented the change in predicted defaults under a static 0.50 cutoff, not mean probability shifts.
5. **Feature Engineering Null Result**: Engineering 8 domain features (ratios, delinquency interaction indices, missingness indicators) yielded **no improvement** in holdout ROC-AUC across tree models ($\Delta\text{AUC} \le 0$, confirmed by 1,000 bootstrap iterations).
6. **Column-Ordering Bug Fixed in TreeSHAP**: The original SHAP pipeline misaligned feature names due to `ColumnTransformer` column reordering. Restoring exact column alignment confirms `RevolvingUtilizationOfUnsecuredLines` is the dominant predictor (mean $|\text{SHAP}| = 0.72$), followed by `NumberOfTime30-59DaysPastDueNotWorse` ($0.30$) and `age` ($0.25$). LightGBM and XGBoost share high game-theoretic agreement (Spearman $\rho = 0.9394, p = 5.5 \times 10^{-5}$).
7. **Modest Bayesian Tuning Gains**: Optuna TPE optimization over 50 trials yielded modest gains over a standard 48-evaluation Grid Search ($+0.0001$ to $+0.0011$ AUC), disproving claims of dramatic hyperparameter sensitivity.
8. **Classical Basel Scorecard Benchmark**: A Weight-of-Evidence (WOE) quantile-binned Logistic Scorecard achieved ROC-AUC $0.8280$ and Gini $0.6561$, outperforming raw continuous Logistic Regression ($0.8227$) and SVM ($0.8205$).

---

## 1. Comprehensive Numerical Comparison: Old Paper vs. Corrected v2

### Table IV: Baseline Model Performance on Holdout Test Set ($N = 30,000$)

| Model | Metric | Old Paper Table IV | Corrected Holdout ($\tau = 0.50$) | Corrected Holdout (Optimal $\tau^*$) | Optimal $\tau^*$ | Youden $J$ $\tau$ | Discrepancy Cause & Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **LightGBM** | Accuracy | 0.8643 | 0.8643 | 0.8841 | 0.60 | 0.51 | Exact match at $\tau=0.50$. Optimal $\tau$ balances F1. |
| | Precision | 0.2878 | 0.2878 | 0.3108 | | | |
| | Recall | 0.7712 | 0.7712 | 0.6869 | | | |
| | F1-Score | 0.4190 | 0.4190 | **0.4285** | | | |
| | ROC-AUC | **0.8667** | **0.8667** | **0.8667** | — | — | Exact reproduction ($0.0000$ error). |
| **XGBoost** | Accuracy | 0.8621 | 0.8621 | 0.8804 | 0.58 | 0.50 | Exact match at $\tau=0.50$. |
| | Precision | 0.2839 | 0.2839 | 0.3068 | | | |
| | Recall | 0.7707 | 0.7707 | 0.6894 | | | |
| | F1-Score | 0.4149 | 0.4149 | **0.4247** | | | |
| | ROC-AUC | **0.8664** | **0.8664** | **0.8664** | — | — | Exact reproduction ($0.0000$ error). |
| **Random Forest** | Accuracy | 0.8584 | 0.8584 | 0.8785 | 0.59 | 0.51 | Exact match at $\tau=0.50$. |
| | Precision | 0.2748 | 0.2748 | 0.3033 | | | |
| | Recall | 0.7588 | 0.7588 | 0.6690 | | | |
| | F1-Score | 0.4035 | 0.4035 | **0.4172** | | | |
| | ROC-AUC | 0.8639 | 0.8639 | 0.8639 | — | — | Exact reproduction ($0.0000$ error). |
| **Decision Tree** | Accuracy | 0.8524 | 0.8524 | 0.8606 | 0.53 | 0.49 | Exact match at $\tau=0.50$. |
| | Precision | 0.2458 | 0.2458 | 0.2514 | | | |
| | Recall | 0.6874 | 0.6874 | 0.6550 | | | |
| | F1-Score | 0.3621 | 0.3621 | **0.3637** | | | |
| | ROC-AUC | 0.8437 | 0.8437 | 0.8437 | — | — | Exact reproduction ($0.0000$ error). |
| **MLP** | Accuracy | 0.8012 | 0.8012 | 0.8752 | 0.66 | 0.51 | Exact match at $\tau=0.50$. Optimal $\tau$ raises F1 to $0.3636$. |
| | Precision | 0.1983 | 0.1983 | 0.2711 | | | |
| | Recall | 0.7107 | 0.7107 | 0.5516 | | | |
| | F1-Score | 0.3102 | 0.3102 | **0.3636** | | | |
| | ROC-AUC | 0.8376 | 0.8376 | 0.8376 | — | — | Exact reproduction ($0.0000$ error). |
| **Logistic Reg.** | Accuracy | 0.7765 | 0.7765 | 0.8710 | 0.65 | 0.49 | Exact match at $\tau=0.50$. Optimal $\tau$ raises F1 to $0.3475$. |
| | Precision | 0.1772 | 0.1772 | 0.2573 | | | |
| | Recall | 0.7097 | 0.7097 | 0.5352 | | | |
| | F1-Score | 0.2837 | 0.2837 | **0.3475** | | | |
| | ROC-AUC | 0.8227 | 0.8227 | 0.8227 | — | — | Exact reproduction ($0.0000$ error). |
| **Linear SVM** | Accuracy | 0.9332 | 0.9332 | 0.9163 | **0.12** | 0.08 | **Severe threshold artifact explained!** |
| | Precision | 0.4728 | 0.4728 | 0.3967 | | | Platt scaling on CalibratedClassifierCV |
| | Recall | **0.1523** | **0.1523** | **0.4568** | | | pushed probabilities toward base rate ($6.7\%$). |
| | F1-Score | 0.2304 | 0.2304 | **0.4246** | | | At $\tau^*=0.12$, F1 almost doubles ($0.2304 \to 0.4246$). |
| | ROC-AUC | 0.8205 | 0.8205 | 0.8205 | — | — | Exact reproduction ($0.0000$ error). |

---

### Additional Rigor Metrics (Supplementary Table IV-B)

| Model | ROC-AUC | Gini Index ($2 \times \text{AUC} - 1$) | PR-AUC / Avg Precision | KS Statistic | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LightGBM** | 0.8667 | 0.7335 | 0.3957 | 0.5843 | 0.1190 |
| **XGBoost** | 0.8664 | 0.7328 | 0.3937 | 0.5833 | 0.1207 |
| **Random Forest** | 0.8639 | 0.7278 | 0.3846 | 0.5779 | 0.1219 |
| **Decision Tree** | 0.8437 | 0.6874 | 0.3347 | 0.5369 | 0.1378 |
| **MLP** | 0.8376 | 0.6752 | 0.3168 | 0.5284 | 0.1697 |
| **WOE Scorecard** | 0.8280 | 0.6561 | 0.3012 | 0.5102 | 0.1834 |
| **Logistic Regression** | 0.8227 | 0.6454 | 0.2882 | 0.4990 | 0.1866 |
| **Linear SVM** | 0.8205 | 0.6410 | 0.2885 | 0.4947 | 0.0543 |

---

### Statistical Hypothesis Testing: DeLong & Bootstrap (Table VI Revised)

| Comparison Pair | $\Delta$ Test ROC-AUC | DeLong $Z$-score | DeLong $p$-value | Paired Bootstrap $p$-value | Bootstrap 95% CI of $\Delta\text{AUC}$ | Statistical Significance ($\alpha = 0.05$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM vs. XGBoost** | $+0.00029$ | $0.6160$ | **$0.53791$** | **$0.52800$** | $[-0.00063, +0.00124]$ | **NOT SIGNIFICANT** ($p > 0.05$) |
| **LightGBM vs. Random Forest** | $+0.00282$ | $4.4981$ | **$< 0.00001$** | **$< 0.00100$** | $[+0.00159, +0.00405]$ | **SIGNIFICANT** ($p < 0.001$) |
| **XGBoost vs. Random Forest** | $+0.00253$ | $3.9351$ | **$0.00008$** | **$< 0.00100$** | $[+0.00127, +0.00381]$ | **SIGNIFICANT** ($p < 0.001$) |

#### 1,000-Iteration Bootstrap 95% Confidence Intervals:
- **LightGBM**: $0.8667$ $[0.8576, 0.8756]$
- **XGBoost**: $0.8664$ $[0.8573, 0.8753]$
- **Random Forest**: $0.8639$ $[0.8546, 0.8727]$
- **Decision Tree**: $0.8437$ $[0.8336, 0.8532]$
- **MLP**: $0.8376$ $[0.8274, 0.8475]$
- **Logistic Regression**: $0.8227$ $[0.8122, 0.8330]$
- **Linear SVM**: $0.8205$ $[0.8100, 0.8309]$

#### Repeated 5-Fold $\times$ 5-Repeat Cross-Validation (25 Folds on Training Set):
- **LightGBM**: Mean CV AUC = $0.8632 \pm 0.0053$
- **XGBoost**: Mean CV AUC = $0.8628 \pm 0.0052$
- **Random Forest**: Mean CV AUC = $0.8617 \pm 0.0053$

---

### Feature Engineering Evaluation: 10 Features vs. 18 Features (Table VII Revised)

| Model | 10-Feature AUC | 18-Feature AUC | $\Delta$ ROC-AUC | Bootstrap 95% CI of $\Delta\text{AUC}$ | Paired $p$-value | Empirical Conclusion |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **LightGBM** | **0.8667** | 0.8648 | **$-0.0019$** | $[-0.0031, -0.0007]$ | $0.0010$ | **Performance degraded** (statistically significant drop). |
| **XGBoost** | **0.8664** | 0.8656 | **$-0.0008$** | $[-0.0018, +0.0002]$ | $0.1060$ | **Null result** (statistically indistinguishable). |
| **Random Forest** | **0.8639** | 0.8636 | **$-0.0003$** | $[-0.0010, +0.0004]$ | $0.3660$ | **Null result** (statistically indistinguishable). |

*Note*: While holdout ROC-AUC showed a null result, Random Forest demonstrated modest improvements in holdout Precision ($0.2748 \to 0.3015$) and F1-score ($0.4035 \to 0.4136$) at the default threshold, reflecting slightly sharpened probability calibration in high-risk tails.

---

### Macroeconomic Stress Testing: Corrected Logic & Multi-Scenario Audit (Table VIII Revised)

#### The Methodological Flaw:
The original code compounded two shocks:
$$\text{Income Shock: } I_{\text{shocked}} = I_0 \times (1 - 0.20) = 0.80 \, I_0$$
$$\text{Debt Shock: } D_{\text{shocked}} = D_0 \times (1 + 0.20) = 1.20 \, D_0$$
$$\text{DebtRatio}_{\text{shocked}} = \frac{1.20 \, D_0}{0.80 \, I_0} = 1.50 \, \text{DebtRatio}_0 \quad (+50\% \text{ compound})$$
In contrast, the paper text claimed only that *"income drops by 20% each month"*. If debt obligations remain fixed in nominal terms ($D_t = D_0$), a $20\%$ drop in income **mechanically** inflates the debt-to-income ratio by:
$$\Delta \text{DebtRatio} = \frac{D_0}{0.80 \, I_0} = 1.25 \, \text{DebtRatio}_0 \quad (+25\% \text{ analytical increase})$$
Applying both was an unintentional double-shock totaling $+56.25\%$ inflation. Furthermore, the paper text described income falling "each month", yet the implementation applied a single static cross-sectional shock.

#### Reconciling the Paper's $+6.09\%$ and $+6.14\%$ Figures:
In Table VIII, the paper reported $+6.09\%$ (LightGBM) and $+6.14\%$ (XGBoost). Our empirical audit reveals that these values did **not** represent the relative change in mean predicted probability:
- **Baseline predicted default share at $\tau=0.50$**: LightGBM = $17.52\%$, XGBoost = $17.44\%$
- **Shocked predicted default share at $\tau=0.50$**: LightGBM = $23.61\%$, XGBoost = $23.58\%$
- **Absolute difference in default share**: LightGBM = $+6.09\%$, XGBoost = $+6.14\%$
The original paper conflated the **absolute percentage point shift in flagged defaults** at the 0.50 cutoff with the change in expected portfolio default probability. The true **relative increase in mean predicted probability** under the corrected static macroeconomic shock is $+3.76\%$ for LightGBM and $+3.51\%$ for XGBoost.

#### Corrected Multi-Scenario Stress Test Results:

| Scenario Description | Model | Baseline Mean Prob | Shocked Mean Prob | Relative % Shift in Mean Prob | Flagged Defaults ($\tau=0.50$) | Retained Test AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Scenario 1: Income Contraction Only** | LightGBM | 0.2241 | 0.2263 | $+0.99\%$ | $17.52\% \to 17.89\%$ | 0.8649 |
| ($-20\%$ income, DebtRatio unadjusted) | XGBoost | 0.2255 | 0.2274 | $+0.84\%$ | $17.44\% \to 17.75\%$ | 0.8651 |
| **Scenario 2: Debt Shock Only** | LightGBM | 0.2241 | 0.2303 | $+2.75\%$ | $17.52\% \to 18.66\%$ | 0.8552 |
| ($+25\%$ DebtRatio, income fixed) | XGBoost | 0.2255 | 0.2315 | $+2.65\%$ | $17.44\% \to 18.73\%$ | 0.8548 |
| **Scenario 3: Consistent Macroeconomic Shock** | LightGBM | 0.2241 | 0.2325 | **$+3.76\%$** | $17.52\% \to 19.04\%$ | **0.8530** |
| ($-20\%$ income + analytical $+25\%$ DebtRatio) | XGBoost | 0.2255 | 0.2334 | **$+3.51\%$** | $17.44\% \to 19.09\%$ | **0.8524** |
| **Scenario 4: Severe Crisis Stress** | LightGBM | 0.2241 | 0.2452 | $+9.41\%$ | $17.52\% \to 21.68\%$ | 0.8351 |
| ($-35\%$ income + $+53.8\%$ DebtRatio) | XGBoost | 0.2255 | 0.2468 | $+9.44\%$ | $17.44\% \to 21.84\%$ | 0.8340 |
| **Scenario 5: Credit Line Contraction** | LightGBM | 0.2241 | 0.2418 | $+7.89\%$ | $17.52\% \to 20.32\%$ | 0.8427 |
| ($+20\%$ RevolvingUtilization shock) | XGBoost | 0.2255 | 0.2441 | $+8.25\%$ | $17.44\% \to 20.48\%$ | 0.8415 |

---

### Hyperparameter Tuning: Optuna TPE vs. Grid Search vs. Baselines (Table V Revised)

| Model | Grid Search CV AUC | Grid Search Test AUC | Optuna TPE CV AUC | Optuna TPE Test AUC | Optuna $\Delta\text{Test AUC}$ | Best Hyperparameters (Optuna TPE) |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **LightGBM** | 0.8649 | 0.8670 | **0.8656** | **0.8671** | $+0.0001$ | `lr=0.0205, num_leaves=15, n_est=350, min_child=43, alpha=8.13, lambda=0.53` |
| **XGBoost** | 0.8643 | 0.8661 | **0.8653** | **0.8669** | $+0.0008$ | `lr=0.0318, max_depth=4, n_est=350, sub=0.95, col=0.57, gamma=0.80, alpha=0.004, lambda=4.27` |
| **Random Forest** | 0.8626 | 0.8642 | **0.8636** | **0.8653** | $+0.0011$ | `n_est=200, max_depth=12, min_split=76, min_leaf=45, max_feat='sqrt'` |
| **Decision Tree** | — | — | 0.8502 | 0.8525 | — | `max_depth=7, min_split=98, min_leaf=50, crit='entropy'` |
| **MLP** | — | — | 0.8321 | 0.8344 | — | `alpha=0.0001, lr=0.0020, hidden=(128, 64)` |
| **Logistic Reg.** | — | — | 0.8189 | 0.8226 | — | `C=0.0680, penalty='l2'` |
| **Linear SVM** | — | — | 0.8166 | 0.8209 | — | `C=0.0001, penalty='l2'` |
| **WOE Scorecard** | — | — | — | 0.8280 | — | 8 quantile bins, Laplace smoothed WOE, balanced LogReg |

#### Finding on Bayesian Tuning Margin:
The margin of Optuna TPE over Grid Search is modest ($+0.0001$ for LightGBM, $+0.0008$ for XGBoost, $+0.0011$ for RF). Standard grid search over reasonable parameter grids captures the vast majority of tree ensemble capacity on tabular credit data.

---

### Computational Latency & System Footprint (Table IX Revised)

| Model | Training Latency (s) | Inference Latency (ms / 1k rows) | Peak RAM Allocation (MB) |
| :--- | :---: | :---: | :---: |
| **Decision Tree** | 1.15 | 0.32 | 185 |
| **Linear SVM** | 11.42 | 0.65 | 210 |
| **Logistic Regression** | 2.85 | 0.48 | 195 |
| **LightGBM** | **2.62** | **1.84** | **240** |
| **XGBoost** | 6.84 | 4.12 | 290 |
| **Random Forest** | 28.50 | 12.80 | 380 |
| **MLP** | 35.10 | 2.15 | 260 |

---

## 2. TreeSHAP Explainability & Column Alignment Audit

### Column-Ordering Bug Resolution:
In `sklearn.compose.ColumnTransformer`, features are emitted strictly in the order of the transformer execution list:
1. Median-imputed: `[MonthlyIncome, age]`
2. Mode-imputed: `[NumberOfDependents]`
3. Remainder/passthrough: `[RevolvingUtilizationOfUnsecuredLines, NumberOfTime30-59DaysPastDueNotWorse, DebtRatio, NumberOfOpenCreditLinesAndLoans, NumberOfTimes90DaysLate, NumberRealEstateLoansOrLines, NumberOfTime60-89DaysPastDueNotWorse]`

In the original pipeline script, the raw feature list `FEATURES` was indexed directly against the transformed matrix. This created a column misalignment in the generated SHAP summary figures. By dynamically extracting column names via `preprocessor.get_feature_names_out()`, the true game-theoretic ranking was restored.

### Global TreeSHAP Feature Ranking (Holdout Test Set):

| Rank | Feature | LightGBM Mean $|\text{SHAP}|$ | XGBoost Mean $|\text{SHAP}|$ | Spearman Agreement |
| :---: | :--- | :---: | :---: | :---: |
| **1** | `RevolvingUtilizationOfUnsecuredLines` | **0.7226** | **0.7184** | Identical #1 |
| **2** | `NumberOfTime30-59DaysPastDueNotWorse` | **0.3045** | **0.3012** | Identical #2 |
| **3** | `age` | **0.2514** | **0.2488** | Identical #3 |
| **4** | `NumberOfTimes90DaysLate` | **0.2310** | **0.2345** | Identical #4 |
| **5** | `DebtRatio` | 0.1845 | 0.1892 | High |
| **6** | `NumberOfTime60-89DaysPastDueNotWorse` | 0.1420 | 0.1385 | High |
| **7** | `MonthlyIncome` | 0.1150 | 0.1190 | High |
| **8** | `NumberOfOpenCreditLinesAndLoans` | 0.0890 | 0.0865 | High |
| **9** | `NumberRealEstateLoansOrLines` | 0.0512 | 0.0530 | High |
| **10** | `NumberOfDependents` | 0.0385 | 0.0372 | High |

- **Cross-Architecture Game-Theoretic Concordance**: Spearman rank correlation $\rho = 0.9394$ ($p = 5.5 \times 10^{-5}$). Both architectures rely on the identical core risk hierarchy.

---

## 3. Paper Claims Requiring Formal Revision or Retraction

1. **Claim: "LightGBM statistically outperforms XGBoost and represents the premier architecture."**  
   - **Correction**: **Retract claim of statistical superiority.** DeLong test $p = 0.53791$ proves LightGBM ($0.8667$) and XGBoost ($0.8664$) are statistically equivalent on this dataset. State instead that both models represent an identical top-tier performance cluster, with LightGBM offering faster training ($2.62\text{s}$ vs $6.84\text{s}$) and lower inference latency ($1.84\text{ms}$ vs $4.12\text{ms}$).
2. **Claim: "LightGBM exhibits superior resilience under macroeconomic stress compared to XGBoost."**  
   - **Correction**: **Retract claim.** Under the corrected, economically consistent stress shock, XGBoost's mean predicted default probability increases by $+3.51\%$, while LightGBM's increases by $+3.76\%$. Under shock, LightGBM retains $0.8530$ AUC and XGBoost retains $0.8524$ AUC. The difference is negligible.
3. **Claim: "Domain-specific feature engineering yields significant discrimination gains."**  
   - **Correction**: **Retract claim of AUC gains.** Across all gradient boosted trees, adding 8 engineered interaction and ratio features resulted in a null or slightly negative change in test ROC-AUC ($\Delta\text{AUC} = -0.0019$ for LightGBM, $-0.0008$ for XGBoost). Note that while AUC is unchanged, Random Forest experienced modest precision gains ($0.2748 \to 0.3015$) in extreme percentiles.
4. **Claim: "Linear SVM is intrinsically unsuitable for credit default scoring due to a 15.2% recall."**  
   - **Correction**: **Reframe as a decision threshold artifact.** The low recall was caused by applying the arbitrary $0.50$ cutoff to Platt-scaled probabilities on an imbalanced dataset. Calibrating the decision threshold to $\tau^* = 0.12$ yields $45.68\%$ recall and an F1-score of $0.4246$, demonstrating that linear hyperplanes with calibrated cutoffs are competitive.
5. **Claim: "Bayesian optimization via TPE achieves dramatic performance leaps over traditional Grid Search."**  
   - **Correction**: **Moderate the claim.** TPE achieved an AUC advantage of only $+0.0001$ to $+0.0011$ over standard 48-point grid search. Emphasize TPE's parameter-efficiency in searching continuous spaces rather than a massive accuracy margin.

---

## 4. Camera-Ready Replacement Text for the Paper

### A. Abstract Replacement Text
> *"Credit default prediction under extreme class imbalance demands rigorous evaluation beyond default classification cutoffs. This study benchmarks seven machine learning models on the 150,000-observation Give Me Some Credit dataset using strict anti-leakage cross-validation. Gradient boosted decision trees achieved top-tier discrimination, with LightGBM (ROC-AUC 0.8667, 95% CI [0.8576, 0.8756]) and XGBoost (ROC-AUC 0.8664, 95% CI [0.8573, 0.8753]) exhibiting statistically indistinguishable performance (DeLong test $p = 0.538$). Crucially, we demonstrate that standard 0.50 decision cutoffs produce severe evaluation distortions in imbalanced credit portfolios: threshold optimization via out-of-fold F1-maximization elevates linear SVM recall from an apparent 15.2% to 45.7% (F1: 0.4246), rivaling ensemble architectures. Macroeconomic stress testing under consistent nominal debt dynamics reveals that a 20% income contraction inflates expected default probabilities by 3.76% (LightGBM) and 3.51% (XGBoost), with discrimination remaining robust above 0.852 AUC. Finally, game-theoretic TreeSHAP interpretability reveals near-perfect architectural concordance (Spearman $\rho = 0.939, p < 10^{-4}$), identifying revolving credit utilization and short-term delinquency as the primary drivers of portfolio distress."*

---

### B. Results & Discussion Replacement Text (Section IV & V)
> *"**Comparative Discrimination & Hypothesis Testing**: As detailed in Table IV, tree ensembles dominate linear and neural baselines. LightGBM attained the highest holdout ROC-AUC (0.8667), closely followed by XGBoost (0.8664) and Random Forest (0.8639). However, formal hypothesis testing demonstrates that the observed 0.00029 AUC difference between LightGBM and XGBoost is not statistically significant (DeLong $Z = 0.6160, p = 0.538$; paired bootstrap $p = 0.528$). In contrast, both gradient boosting architectures significantly outperform Random Forest ($p < 0.001$). LightGBM maintains a distinct operational advantage in computational efficiency, executing training in 2.62 s compared to 6.84 s for XGBoost, with an inference latency of 1.84 ms per 1,000 observations."*
>
> *"**Threshold Calibration Dynamics**: Credit default datasets with low prevalence (6.68%) inherently depress posterior probability outputs. Evaluated at the default $\tau = 0.50$ cutoff, Linear SVM achieved an apparently dismal recall of 0.1523. When decision thresholds are calibrated via training-set out-of-fold cross-validation ($\tau^* = 0.12$), SVM recall expands to 0.4568 and F1-score rises to 0.4246, surpassing baseline Logistic Regression and Decision Trees. Similar F1 gains are observed across all models when tuned to optimal thresholds ($\tau^* \in [0.53, 0.60]$ for tree ensembles)."*
>
> *"**Macroeconomic Stress Testing & Resilience**: To simulate economic downturns without compounding accounting distortions, we modeled a 20% contraction in MonthlyIncome while holding nominal liabilities constant, inducing an analytical 25% inflation in DebtRatio ($1/0.80 = 1.25$). Under this macroeconomic shock, portfolio mean default probability rose by +3.76% under LightGBM and +3.51% under XGBoost, retaining robust discriminative capacity (AUC 0.8530 and 0.8524, respectively). Rather than demonstrating divergent resilience, both tree architectures exhibit near-identical stability across severity sweeps from -5% to -40% income contraction."*

---

### C. Conclusion Replacement Text (Section VII)
> *"This empirical evaluation demonstrates that while gradient boosting algorithms (LightGBM and XGBoost) provide state-of-the-art discrimination and robustness in consumer credit scoring, their performance is statistically indistinguishable under rigorous testing. Methodologically, our findings caution against evaluating imbalanced credit scoring models using naive 0.50 classification thresholds or compounding stress testing shocks without structural accounting consistency. Furthermore, domain-engineered feature interactions failed to improve tree ensemble discrimination, confirming that modern boosting algorithms internally capture nonlinear risk interactions directly from raw bureau variables. TreeSHAP analysis verifies that both top-performing architectures align with credit risk fundamentals, ensuring compliance with regulatory interpretability standards under Basel III frameworks."*

---

## 5. Artifact Inventory

All generated artifacts adhere strictly to IEEE publication standards (vector PDF and 300 dpi PNG, colorblind-safe palettes, font size $\ge 8$ pt).

### Generated CSV Tables (`outputs_v2/tables/`):
- `repro_check.csv`: Exact baseline replication check.
- `table_iv_v2.csv`: Complete performance metrics at $\tau=0.50$ and optimal OOF thresholds.
- `thresholds.csv`: Threshold optimization summary and SVM comparison.
- `extra_metrics.csv`: Gini, PR-AUC, KS statistic, Brier score.
- `bootstrap_ci.csv`: 1,000-sample bootstrap 95% confidence intervals.
- `significance_tests.csv`: DeLong and paired bootstrap $p$-values.
- `repeated_cv_results.csv`: 25-fold repeated CV mean and standard deviations.
- `stress_test_scenarios.csv`: All 5 stress testing scenarios.
- `stress_test_severity_sweep.csv`: Sensitivity sweeps across income contraction rates.
- `engineered_features.csv`: Definition and rationale for all 8 engineered features.
- `feature_engineering_comparison.csv`: 10 vs. 18 feature empirical comparison.
- `shap_rankings.csv`: TreeSHAP global feature importance and Spearman correlation.
- `runtime_benchmark.csv`: Training and inference latency per 1,000 observations.
- `search_spaces.csv`: Hyperparameter search bounds matching Table III.
- `best_params.csv`: Optimal hyperparameters selected by Optuna TPE.
- `tuning_comparison.csv`: Optuna TPE vs. Grid Search empirical comparison.
- `scorecard_metrics.csv`: Traditional WOE Scorecard baseline metrics.
- `related_work.csv`: Structured literature table replacing deprecated Fig 1.

### Generated Figures (`outputs_v2/figures/`):
- `fig2_pipeline_workflow.pdf` / `.png`
- `fig3_class_distribution.pdf` / `.png`
- `fig4_confusion_matrices_v2.pdf` / `.png`
- `fig5_roc_curves_v2.pdf` / `.png`
- `fig6_stress_test_v2.pdf` / `.png`
- `fig7_shap_summary_beeswarm_v2.pdf` / `.png`
- `fig8_shap_bar_chart_v2.pdf` / `.png`
- `precision_recall_curves_v2.pdf` / `.png`
- `calibration_curves_top3.pdf` / `.png`
- `feature_engineering_roc_v2.pdf` / `.png`
- `shap_dependence_top4_v2.pdf` / `.png`
- `shap_waterfall_high_risk_v2.pdf` / `.png`
- `shap_waterfall_low_risk_v2.pdf` / `.png`
```

---

## 11. Exhaustive Inventory of Irreproducible Documented Numbers

Below is the definitive catalog of every documented numerical value in `CONFERENCE_PAPER_DOCUMENTATION.md` and `CHANGES.md` that cannot be reproduced from the underlying data and pipeline:

1. **Partition Specification ($N_{\text{train}} = 120,000, N_{\text{test}} = 30,000$, 2,005 defaults)**:
   - Ground truth: Stratified 70:30 split ($N_{\text{train}} = 105,000$, $N_{\text{test}} = 45,000$, $n_{\text{defaults}} = 3,008$).
2. **Table IV Holdout Metrics at $\tau = 0.50$**:
   - LightGBM: Documented `Acc=0.8643, Prec=0.2878, Rec=0.7712, F1=0.4190` $\implies$ True: `Acc=0.7912, Prec=0.2126, Rec=0.7852, F1=0.3346`.
   - XGBoost: Documented `Acc=0.8621, Prec=0.2839, Rec=0.7707, F1=0.4149` $\implies$ True: `Acc=0.7953, Prec=0.2149, Rec=0.7773, F1=0.3367`.
   - Random Forest: Documented `Acc=0.8584, Prec=0.2748, Rec=0.7588, F1=0.4035` $\implies$ True: `Acc=0.8007, Prec=0.2179, Rec=0.7656, F1=0.3393`.
   - Decision Tree: Documented `Acc=0.8524, Prec=0.2458, Rec=0.6874, F1=0.3621` $\implies$ True: `Acc=0.8131, Prec=0.2251, Rec=0.7354, F1=0.3447`.
   - MLP: Documented `Acc=0.8012, Prec=0.1983, Rec=0.7107, F1=0.3102` $\implies$ True: `Acc=0.8129, Prec=0.2205, Rec=0.7098, F1=0.3365`.
   - Logistic Regression: Documented `Acc=0.7765, Prec=0.1772, Rec=0.7097, F1=0.2837` $\implies$ True: `Acc=0.8624, Prec=0.2702, Rec=0.6220, F1=0.3767`.
   - SVM: Documented `Acc=0.9332, Prec=0.4728, F1=0.2304` $\implies$ True: `Acc=0.9363, Prec=0.5902, F1=0.2421`.
3. **Table IV Holdout Metrics at Optimal Threshold $\tau^*$**:
   - LightGBM: Documented `tau*=0.60, Acc=0.8841, Prec=0.3108, Rec=0.6869, F1=0.4285` $\implies$ True: `tau*=0.76, Acc=0.9123, Prec=0.3856, Rec=0.5253, F1=0.4448`.
   - XGBoost: Documented `tau*=0.58, Acc=0.8804, Prec=0.3068, Rec=0.6894, F1=0.4247` $\implies$ True: `tau*=0.79, Acc=0.9203, Prec=0.4167, Rec=0.4801, F1=0.4462`.
   - Random Forest: Documented `tau*=0.59, Acc=0.8785, Prec=0.3033, Rec=0.6690, F1=0.4172` $\implies$ True: `tau*=0.73, Acc=0.9127, Prec=0.3856, Rec=0.5150, F1=0.4410`.
   - Decision Tree: Documented `tau*=0.53, Acc=0.8606, Prec=0.2514, Rec=0.6550, F1=0.3637` $\implies$ True: `tau*=0.79, Acc=0.9158, Prec=0.3940, Rec=0.4817, F1=0.4334`.
   - MLP: Documented `tau*=0.66, Acc=0.8752, Prec=0.2711, Rec=0.5516, F1=0.3636` $\implies$ True: `tau*=0.23, Acc=0.4598, Prec=0.1034, Rec=0.9232, F1=0.1860`.
   - Logistic Regression: Documented `tau*=0.65, Acc=0.8710, Prec=0.2573, Rec=0.5352, F1=0.3475` $\implies$ True: `tau*=0.66, Acc=0.9126, Prec=0.3786, Rec=0.4807, F1=0.4236`.
   - SVM: Documented `Acc=0.9163, Youden=0.08` $\implies$ True: `Acc=0.9172, Youden=0.0647`.
4. **Bootstrap 95% Confidence Intervals for Holdout AUC**:
   - LightGBM: Documented `[0.8576, 0.8756]` $\implies$ True: `[0.8600, 0.8735]`.
   - XGBoost: Documented `[0.8573, 0.8753]` $\implies$ True: `[0.8596, 0.8730]`.
   - Random Forest: Documented `[0.8546, 0.8727]` $\implies$ True: `[0.8571, 0.8704]`.
   - Decision Tree: Documented `[0.8336, 0.8532]` $\implies$ True: `[0.8361, 0.8509]`.
   - MLP: Documented `[0.8274, 0.8475]` $\implies$ True: `[0.8289, 0.8453]`.
   - Logistic Regression: Documented `[0.8122, 0.8330]` $\implies$ True: `[0.8141, 0.8312]`.
   - Linear SVM: Documented `[0.8100, 0.8309]` $\implies$ True: `[0.8119, 0.8291]`.
5. **Additional Credit Metrics**:
   - LightGBM: Documented `PR-AUC=0.3957, KS=0.5843, Brier=0.1190` $\implies$ True: `PR-AUC=0.4022, KS=0.5808, Brier=0.1423`.
   - XGBoost: Documented `PR-AUC=0.3937, KS=0.5833, Brier=0.1207` $\implies$ True: `PR-AUC=0.4029, KS=0.5756, Brier=0.1410`.
   - Random Forest: Documented `PR-AUC=0.3846, KS=0.5779, Brier=0.1219` $\implies$ True: `PR-AUC=0.3933, KS=0.5721, Brier=0.1364`.
   - Decision Tree: Documented `PR-AUC=0.3347, KS=0.5369, Brier=0.1378` $\implies$ True: `PR-AUC=0.3541, KS=0.5541, Brier=0.1483`.
   - MLP: Documented `PR-AUC=0.3168, KS=0.5284, Brier=0.1697` $\implies$ True: `PR-AUC=0.3655, KS=0.5318, Brier=0.1472`.
   - Logistic Regression: Documented `PR-AUC=0.2882, KS=0.4990, Brier=0.1866` $\implies$ True: `PR-AUC=0.3564, KS=0.5119, Brier=0.1548`.
   - Linear SVM: Documented `PR-AUC=0.2885, KS=0.4947, Brier=0.0543` $\implies$ True: `PR-AUC=0.3535, KS=0.5029, Brier=0.0528`.
6. **Stress Testing Numbers**:
   - Baseline Mean Probability: Documented `0.2241` (LGBM) and `0.2255` (XGB) $\implies$ True: `0.3195` (LGBM) and `0.3219` (XGB).
   - Retained Test AUC under Macro Shock: Documented `0.8530` (LGBM) and `0.8524` (XGB) $\implies$ True: `0.8660` (LGBM) and `0.8660` (XGB).
   - Flagged Defaults Shift: Documented `17.52% -> 19.04%` $\implies$ True: `24.69% -> 26.20%` (LGBM) and `24.18% -> 25.66%` (XGB).
7. **SHAP Feature Importance Values**:
   - Feature attributions in Table 11.1 (`0.7226, 0.3045, 0.2514, 0.2310, 0.1845, 0.1420, 0.1150, 0.0890, 0.0512, 0.0385`) do not match the values in `shap_rankings.csv` (`0.717150, 0.303227, 0.246680, 0.228810, 0.159501, 0.121267, 0.120917, 0.096838, 0.092092, 0.010241`).
8. **Runtime Benchmarks**:
   - Latencies in Table 15 (`Train = 2.62 s, Inference = 1.84 ms, RAM = 240 MB` for LightGBM; `6.84 s, 4.12 ms` for XGBoost) do not match `runtime_benchmark.csv` (`Train = 0.48 s, Inference = 1.08 ms` for LightGBM; `0.39 s, 0.57 ms` for XGBoost).
