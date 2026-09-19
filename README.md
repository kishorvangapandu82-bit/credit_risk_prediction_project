# Credit Scoring Research: Machine Learning & Deep Learning Benchmarking
### Dataset: *Give Me Some Credit* (GMSC) | Framework: Scikit-Learn & XGBoost

---

## 📌 Executive Summary

This repository contains an end-to-end, publication-grade credit risk modeling framework adhering strictly to rigorous academic and regulatory standards (e.g., Basel II/III, CCAR). 

We systematically benchmark **four diverse model paradigms** across 150,000 borrower records to predict serious delinquency within a two-year horizon (`SeriousDlqin2yrs`):
1. **Logistic Regression** (Traditional parametric statistical baseline)
2. **Decision Tree (CART)** (Shallow non-parametric rule-based model)
3. **XGBoost** (Extreme Gradient Boosted Decision Trees ensemble)
4. **Multi-Layer Perceptron (MLP)** (Deep feedforward neural network)

---

## 🏆 Final Benchmark Results (Untouched 30% Holdout Test Set, N = 45,000)

All models were tuned strictly using **Stratified 10-Fold Cross-Validation** on the 70% training split with full preprocessing isolation to prevent data leakage.

| Rank | Model Architecture | ROC-AUC | Recall (Class 1) | Precision (Class 1) | F1-Score | Accuracy | Test Set Size |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 | **LightGBM (Gradient Boosting)** | **0.8667** | **0.7852** | 0.2126 | 0.3346 | 0.7912 | 45,000 |
| 🥈 | **XGBoost (Gradient Boosting)** | **0.8664** | **0.7773** | 0.2149 | 0.3367 | 0.7953 | 45,000 |
| 🥉 | **Decision Tree (CART)** | **0.8437** | **0.7354** | 0.2251 | 0.3447 | 0.8131 | 45,000 |
| 4th | **MLP Neural Network** | **0.8376** | **0.7098** | 0.2205 | 0.3365 | 0.8129 | 45,000 |
| 5th | **Logistic Regression (Baseline)** | **0.8227** | **0.6220** | 0.2702 | **0.3767** | 0.8624 | 45,000 |
| 6th | **Support Vector Machine (Linear SVM)** | **0.8205** | 0.1523 | **0.5902** | 0.2421 | **0.9363** | 45,000 |

### 🔑 Key Academic Insights:
1. **Histogram Gradient Boosting (LightGBM & XGBoost)**: Both gradient boosted tree algorithms lead the benchmark, achieving **0.8667** and **0.8664 ROC-AUC**. LightGBM achieves the highest default recall (**78.52%**), catching 2,362 out of 3,008 defaults.
2. **Support Vector Machine (SVM)**: Operating with calibrated margins, Linear SVM achieved **0.8205 ROC-AUC** and the highest precision (**59.02%**), highlighting the fundamental trade-off between conservative classification boundaries and default detection recall.
3. **Model Hierarchy**: Gradient Boosted Trees (LightGBM / XGBoost) > Single Decision Tree (CART) > Feedforward Neural Net (MLP) > Linear Hyperplane (Logistic Regression / SVM).

---

## 🌪️ Macroeconomic Stress Testing Results

We subjected all models to a simulated macroeconomic shock scenario:
- **Income Contraction**: `MonthlyIncome` drops by **20%** (`× 0.80`)
- **Debt Service Increase**: `DebtRatio` rises by **25%** (`× 1.25`)

| Model | Baseline AUC | Stressed AUC | AUC Degradation | Mean Prob Increase | Default Rate Surge |
|:---|:---:|:---:|:---:|:---:|:---:|
| **LightGBM** | **0.8667** | **0.8660** | **+0.0007** | **+3.76%** | **+6.09%** |
| **XGBoost** | **0.8664** | **0.8660** | **+0.0004** | **+3.51%** | **+6.14%** |
| **Decision Tree** | 0.8437 | 0.8438 | -0.0000 | -0.01% | -0.06% |
| **MLP Neural Network** | 0.8376 | 0.8372 | +0.0003 | +3.35% | +4.38% |
| **Logistic Regression** | 0.8227 | 0.8221 | +0.0006 | +0.88% | +1.47% |
| **Support Vector Machine** | 0.8205 | 0.8198 | +0.0007 | +1.22% | +0.13% |

- **Ranking Stability**: All models exhibited **>99.9% AUC retention**, demonstrating that relative risk ranking remains valid during recessions.
- **Capital Sensitivity**: **XGBoost demonstrated the highest economic sensitivity** (+6.14% surge in projected defaults), allowing financial institutions to dynamically raise loan-loss provisions before defaults materialize.

---

## 🔍 Explainable AI (XAI): SHAP Analysis

To satisfy banking regulatory compliance (US Equal Credit Opportunity Act, Fair Lending, FCRA), we applied **TreeSHAP** to our best model (**XGBoost**) to generate both global and local explanations:

| Rank | Feature | Mean \|SHAP\| | Directional Impact on Default Risk |
|:---:|:---|:---:|:---|
| 1 | **`RevolvingUtilizationOfUnsecuredLines`** | **0.8247** | High utilization exponentially increases default likelihood. |
| 2 | **`NumberOfTime30-59DaysPastDueNotWorse`** | **0.3627** | Recent minor delinquency is a strong early-warning signal. |
| 3 | **`NumberOfTimes90DaysLate`** | **0.3231** | Severe delinquency sharply pushes score into high-risk territory. |
| 4 | **`age`** | **0.2322** | Older borrowers consistently exhibit lower default rates. |
| 5 | **`NumberOfTime60-89DaysPastDueNotWorse`** | **0.1796** | Intermediate delinquency indicator. |
| 6 | **`NumberOfOpenCreditLinesAndLoans`** | **0.1622** | Moderate open credit correlates with prime credit profile. |
| 7 | **`DebtRatio`** | **0.1297** | High debt obligations increase default probability. |
| 8 | **`MonthlyIncome`** | **0.1113** | Higher income provides a safety buffer against default. |
| 9 | **`NumberRealEstateLoansOrLines`** | **0.1009** | Asset-backed credit indicator. |
| 10 | **`NumberOfDependents`** | **0.0186** | Minimal marginal impact on baseline default probability. |

---

## 📊 Visualizations Generated (`plots/`)

All figures were rendered at publication-quality 300 DPI:

### Model Comparison & Stress Testing:
1. `plots/roc_curves_comparison.png`: Combined ROC Curves with AUC annotations across all 4 models.
2. `plots/precision_recall_comparison.png`: Precision-Recall Curves demonstrating performance over the 6.68% minority baseline.
3. `plots/calibration_curves.png`: Reliability diagrams evaluating how accurately predicted probabilities match observed empirical default rates.
4. `plots/metrics_comparison_barchart.png`: Multi-metric grouped comparative bar chart.
5. `plots/confusion_matrices.png`: 2x2 grid showing True Positives, True Negatives, False Positives, and False Negatives for each model.
6. `plots/stress_test_impact.png`: Two-panel evaluation of AUC resilience and risk score migration under macroeconomic distress.

### Explainable AI (SHAP):
7. `plots/shap_summary_beeswarm.png`: Global feature impact & directionality summary plot.
8. `plots/shap_feature_importance_bar.png`: Global Mean |SHAP| feature ranking bar chart.
9. `plots/shap_dependence_revolving_util.png`: Non-linear dependence curve for Revolving Utilization.
10. `plots/shap_waterfall_high_risk.png`: Individual local explanation (Adverse Action Notice reasons for rejected applicant).
11. `plots/shap_waterfall_low_risk.png`: Individual local explanation (Credit Approval justification for prime applicant).

---

## 📁 Repository Structure

```
credit-scoring-project/
├── data/
│   └── cs-training.csv                   # Raw Give Me Some Credit dataset (150k rows)
├── models/
│   ├── logistic_regression.joblib       # Serialized tuned Logistic Regression pipeline
│   ├── decision_tree.joblib             # Serialized tuned Decision Tree pipeline
│   ├── mlp.joblib                       # Serialized tuned MLP Neural Network pipeline
│   └── xgboost.joblib                   # Serialized tuned XGBoost pipeline
├── plots/
│   ├── roc_curves_comparison.png
│   ├── precision_recall_comparison.png
│   ├── calibration_curves.png
│   ├── metrics_comparison_barchart.png
│   ├── confusion_matrices.png
│   └── stress_test_impact.png
├── results/
│   ├── logistic_regression_metrics.csv
│   ├── decision_tree_metrics.csv
│   ├── mlp_metrics.csv
│   ├── xgboost_metrics.csv
│   └── stress_test_comparison.csv
├── src/
│   ├── config.py                        # Central project parameters and constants
│   ├── data_preprocessing.py            # Imputation, artifact capping & leakage-safe pipeline
│   ├── evaluation.py                    # Multi-metric evaluation and reporting module
│   ├── models.py                        # Scikit-learn Pipeline factories for all 4 models
│   ├── split_verify.py                  # Stratified train/test split verification
│   ├── train_logistic.py                # Logistic Regression 10-fold CV training script
│   ├── train_decision_tree.py           # Decision Tree 10-fold CV training script
│   ├── train_xgboost.py                 # XGBoost 10-fold CV training script
│   ├── train_mlp.py                     # MLP Neural Network 10-fold CV training script
│   ├── visualize_results.py             # 300 DPI comparative plotting script
│   ├── stress_testing.py                # Macroeconomic stress testing engine
│   └── shap_analysis.py                 # TreeSHAP explainability engine
├── TRAINING_DETAILS.md                  # Detailed epochs, boosting rounds & convergence specs
├── PERFORMANCE_OPTIMIZATION_ROADMAP.md  # Feature engineering & ensemble optimization guide
└── README.md                            # Comprehensive project report & documentation
```

---

## 🚀 How to Reproduce

To reproduce the entire pipeline from scratch:

```bash
cd credit-scoring-project

# 1. Verify data cleaning and stratified splitting
python src/split_verify.py

# 2. Train and tune all 4 models (10-fold CV)
python src/train_logistic.py
python src/train_decision_tree.py
python src/train_xgboost.py
python src/train_mlp.py

# 3. Generate all publication figures
python src/visualize_results.py

# 4. Run macroeconomic stress test
python src/stress_testing.py
```
