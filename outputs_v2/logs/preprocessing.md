# Rigorous Preprocessing & Anti-Leakage Protocol Documentation

**Project**: An Empirical Evaluation of Machine Learning Models for Credit Default Prediction: Financial Stress Testing and SHAP-Based Interpretability  
**Dataset**: Kaggle *Give Me Some Credit* (GMSC), $N = 150,000$ borrower profiles  
**Target Variable**: `SeriousDlqin2yrs` (Binary default indicator; default rate = 6.6838%)  
**Partitioning Protocol**: Stratified 70:30 split ($N_{\text{train}} = 105,000$, $N_{\text{test}} = 45,000$, seed = 42)

---

## 1. Systematic Handling of Institutional Encoding Artifacts (Codes 96 and 98)

In the Give Me Some Credit (GMSC) empirical dataset, three delinquency count covariates exhibit clusters of exact integer values taking **96** and **98**:
1. `NumberOfTime30-59DaysPastDueNotWorse`
2. `NumberOfTimes90DaysLate`
3. `NumberOfTime60-89DaysPastDueNotWorse`

### Behavioral & Institutional Context
Values 96 and 98 are legacy institutional credit bureau data capture codes used to denote:
- Code 96: Indeterminate or uncontactable borrower / data unavailable.
- Code 98: Missing, disputed, or charge-off status without granular monthly bucket tracking.

If treated as continuous or count variables, parametric models (Logistic Regression, Neural Networks) and distance-based algorithms (SVMs) treat a borrower with code 98 as having defaulted 98 separate times over a 24-month observation window—a mathematical impossibility that severely distorts gradients and inflates loss penalties.

### Capping Implementation
- Artifact values ($\in \{96, 98\}$) are identified across all three delinquency columns.
- The 99th percentile of **valid observations** (excluding 96 and 98) is computed strictly on the training partition ($N = 105,000$).
- For each affected column, all occurrences of 96 and 98 are capped at this 99th percentile value ($\le 5$ delinquencies).

---

## 2. Missing Value Imputation Protocol

Missing values are present in two primary attributes:
- `MonthlyIncome`: Missing in **29,731 records (19.82%)**.
- `NumberOfDependents`: Missing in **3,924 records (2.62%)**.
- In addition, biologically impossible values in `age` ($age = 0$, $N = 1$) are converted to `NaN`.

### Anti-Leakage Imputation Strategy
All imputation parameters are estimated **strictly on the 105,000 training records** and stored within an `sklearn.compose.ColumnTransformer`:
1. **`MonthlyIncome`**: Imputed using the **training median** ($\text{Median}_{\text{train}} = \$5,400.00$). Median imputation is robust to severe positive skewness in retail income distributions.
2. **`age`**: Imputed using the **training median** ($\text{Median}_{\text{train}} = 52.0\text{ years}$).
3. **`NumberOfDependents`**: Imputed using the **training mode / most frequent class** ($\text{Mode}_{\text{train}} = 0.0\text{ dependents}$).

The holdout test partition ($N = 45,000$) is never accessed during parameter estimation; it is transformed using the frozen parameters learned on the training set.

---

## 3. The GMSC Missing-Income & DebtRatio Structural Quirk

A known artifact of the GMSC dataset is that when `MonthlyIncome` is missing, the `DebtRatio` attribute holds the **raw dollar-denominated monthly debt and living expenditure** rather than a true dimensionless ratio ($\text{MonthlyDebt} / \text{MonthlyIncome}$).

### Empirical Consequences
- For borrowers with recorded income, `DebtRatio` is typically in the range $[0.0, 5.0]$.
- For borrowers with missing income, `DebtRatio` spans into extreme values up to **$329,664.0$** (representing absolute dollar payments).
- Uncapped multiplication of `MonthlyIncome` by `DebtRatio` in naive feature engineering creates artificial astronomical debts ($5,400 \times 329,664 \approx \$1.78\text{ billion}$).

### Resolution in our Pipeline
1. **Missing Income Indicator**: An explicit indicator feature (`MissingIncomeFlag = 1`) is constructed prior to imputation to allow non-linear models to isolate unverified income applicants.
2. **Feature Engineering Guardrail**: When computing monetary debt interactions, `DebtRatio` is clamped at its 99th percentile ($10.0$) before multiplying by imputed monthly income.
3. **Stress Testing Guardrail**: In macroeconomic stress scenarios, borrowers with missing income are tracked separately to avoid misinterpreting dollar obligations as leverage ratios.

---

## 4. Extreme Values and Outlier Treatment

1. **`RevolvingUtilizationOfUnsecuredLines`**: Values exceed $1.0$ in $3,321$ records, reaching an extreme upper bound of $50,708.0$. In banking, utilization can legitimately exceed $100\%$ due to over-limit penalties, interest accrual, or negative balances; however, values in the thousands represent data entry anomalies (dollar balance recorded instead of ratio).
   - In baseline tree models, monotonic rank-order splitting isolates extreme outliers without degrading split thresholds.
   - In domain feature engineering, an explicit binary indicator (`OverUtilizationFlag = 1` for utilization $> 1.0$) was engineered.
2. **`DebtRatio`**: Reaches extreme values up to $329,664.0$ due to the missing-income quirk detailed above.

---

## 5. Feature Standardization & Scaling Protocol

- **`StandardScaler`** ($\mu = 0, \sigma = 1$) is fitted strictly on the training partition.
- **Architectural Routing**:
  - Parametric and distance-based architectures (**Logistic Regression, Linear SVM, MLP Neural Network**) strictly require standardized features for convex solver convergence and numerical stability.
  - Tree-based architectures (**Decision Trees, Random Forest, XGBoost, LightGBM**) are invariant to monotonic linear transformations, but are evaluated through identical pipeline wrappers for standardized benchmarking.

---

## 6. Verification of Cross-Validation Anti-Leakage Pipeline

To verify whether preprocessing is refitted inside each cross-validation fold:

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Leakage-safe pipeline construction:
preprocessor = ColumnTransformer(
    transformers=[
        ("median_impute", SimpleImputer(strategy="median"), ["MonthlyIncome", "age"]),
        ("mode_impute", SimpleImputer(strategy="most_frequent"), ["NumberOfDependents"]),
        ("other", SimpleImputer(strategy="median"), [
            "RevolvingUtilizationOfUnsecuredLines",
            "NumberOfTime30-59DaysPastDueNotWorse",
            "DebtRatio",
            "NumberOfOpenCreditLinesAndLoans",
            "NumberOfTimes90DaysLate",
            "NumberRealEstateLoansOrLines",
            "NumberOfTime60-89DaysPastDueNotWorse"
        ]),
    ]
)

full_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("scaler", StandardScaler()),
    ("classifier", model)
])
```

Because `full_pipeline` is passed directly into `cross_val_score`, `cross_val_predict`, `GridSearchCV`, and Optuna evaluation loops, **the `preprocessor` and `scaler` are refitted de novo on each training fold ($K-1$ folds) and applied out-of-fold to the validation fold**. Zero information leaks across CV partitions or from the holdout test set.
