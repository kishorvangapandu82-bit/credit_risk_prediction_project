# Credit Scoring Performance Optimization Roadmap
### Proven Engineering Strategies to Maximize ROC-AUC & Recall

This document details the recommended methodologies for pushing credit scoring model performance from the current **0.8664 ROC-AUC baseline** toward the theoretical ceiling (**0.875+ ROC-AUC**).

---

## 1. Hierarchy of Performance Optimization Techniques

In empirical credit risk research and competitive machine learning, performance gains follow a distinct hierarchy:

```
Methodology ROI:
[Rank 1] Domain-Specific Feature Engineering   ████████████████  (+0.010 to +0.015 AUC)
[Rank 2] Multi-Model Stacking & Blending       ████████  (+0.005 to +0.008 AUC)
[Rank 3] Optimal Decision Threshold Tuning     ████████  (+10% to +15% F1 / Precision)
[Rank 4] Advanced Hyperparameter Optimization  ████  (+0.002 to +0.004 AUC)
```

---

## 2. Strategy #1: Domain-Specific Financial Feature Engineering (Highest ROI)

Tree-based algorithms like XGBoost split parallel to feature axes ($X_i \le \theta$). They cannot naturally construct diagonal interaction lines or non-linear financial ratios unless explicitly provided.

### Recommended Engineered Features:

### A. Delinquency Aggregation & Severity
1. **`TotalPastDueCount`**:
   $$\text{TotalPastDue} = \text{NumberOfTime30-59Days} + \text{NumberOfTime60-89Days} + \text{NumberOfTimes90DaysLate}$$
   *Rationale*: Captures overall delinquency frequency regardless of specific bucket.

2. **`DelinquencySeverityIndex`**:
   $$\text{SeverityIndex} = 1 \cdot (\text{30-59Days}) + 2 \cdot (\text{60-89Days}) + 3 \cdot (\text{90DaysLate})$$
   *Rationale*: Weights severe delinquency (90+ days) three times higher than temporary 30-day delays.

### B. Cash Flow & Solvency Metrics
3. **`IncomePerDependent`**:
   $$\text{IncomePerPerson} = \frac{\text{MonthlyIncome}}{\text{NumberOfDependents} + 1}$$
   *Rationale*: Measures actual discretionary purchasing power per household member.

4. **`EstimatedMonthlyDebt`**:
   $$\text{MonthlyDebtDollars} = \text{MonthlyIncome} \times \text{DebtRatio}$$
   *Rationale*: Converts a unitless ratio into absolute monetary obligations.

5. **`DisposableIncome`**:
   $$\text{DisposableIncome} = \text{MonthlyIncome} - \text{EstimatedMonthlyDebt}$$
   *Rationale*: Represents the liquidity cushion available to absorb emergency financial shocks.

### C. Credit Structure & Behavioral Signals
6. **`UnsecuredLinesRatio`**:
   $$\text{UnsecuredRatio} = \frac{\text{NumberOfOpenCreditLinesAndLoans} - \text{NumberRealEstateLoansOrLines}}{\text{NumberOfOpenCreditLinesAndLoans} + 1}$$
   *Rationale*: Isolates unsecured credit cards from collateralized real estate mortgages.

7. **`MissingIncomeFlag`**:
   $$\text{MissingIncome} = \mathbb{I}(\text{MonthlyIncome is NaN})$$
   *Rationale*: Missing income in credit applications frequently signals self-employment or unverified high-risk borrowers.

8. **`OverUtilizationFlag`**:
   $$\text{OverUtilized} = \mathbb{I}(\text{RevolvingUtilization} > 1.0)$$
   *Rationale*: Flags borrowers actively exceeding credit card limits.

---

## 3. Strategy #2: Model Stacking & Probability Blending

Combining diverse model families cancels out structural blindspots:

### Weighted Probability Blending Formula:
$$\hat{P}_{\text{Final}} = w_{\text{XGB}} \cdot P_{\text{XGBoost}} + w_{\text{MLP}} \cdot P_{\text{MLP}} + w_{\text{DT}} \cdot P_{\text{DecisionTree}} + w_{\text{LR}} \cdot P_{\text{LogReg}}$$

Recommended initial weights:
- $w_{\text{XGB}} = 0.60$ (Dominant non-linear ensemble)
- $w_{\text{MLP}} = 0.20$ (Smooth neural representation)
- $w_{\text{DT}} = 0.10$ (Rule-based boundaries)
- $w_{\text{LR}} = 0.10$ (Linear statistical anchor)

---

## 4. Strategy #3: Decision Threshold Optimization

The standard default threshold is **0.50**. In unbalanced credit scoring (where the default rate is only **6.68%**), a 0.50 threshold is sub-optimal.

### Youden's J-Statistic Optimization:
$$J = \text{Sensitivity} + \text{Specificity} - 1 = \text{TPR} - \text{FPR}$$
- By selecting the threshold that maximizes $J$ (typically between **0.25 and 0.35**), the model achieves balanced default detection without excessive false alarms.

### Cost-Utility Threshold:
$$\text{Cost} = C_{\text{FN}} \cdot \text{FN} + C_{\text{FP}} \cdot \text{FP}$$
- Setting $C_{\text{FN}} \approx 5 \times C_{\text{FP}}$ aligns model decisions with real-world banking profit/loss economics.

---

## 5. Strategy #4: Fine-Grained Hyperparameter Optimization (Optuna)

To squeeze the final +0.002 to +0.004 from XGBoost, tune the regularization subspace using Bayesian optimization:

```python
search_space = {
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9],   # Feature subsampling
    "subsample":        [0.7, 0.8, 0.9, 1.0],   # Row subsampling
    "min_child_weight": [1, 3, 5, 10],          # Leaf inertia
    "gamma":            [0.0, 0.1, 0.5, 1.0],   # Split threshold
    "reg_alpha":        [0.0, 0.1, 1.0, 5.0],   # L1 regularization
    "reg_lambda":       [1.0, 3.0, 5.0, 10.0],  # L2 regularization
}
```
