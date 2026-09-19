"""
=============================================================================
DOMAIN-SPECIFIC FINANCIAL FEATURE ENGINEERING (PHASE 2)
=============================================================================
Project : GMSC Credit Scoring Research
File    : src/feature_engineering.py
Purpose : Construct 8 financial, behavioral, and cash-flow interaction features
          without data leakage.

Features Engineered:
  1. TotalPastDueCount        : (30-59 days) + (60-89 days) + (90+ days)
  2. DelinquencySeverityIndex : 1*(30-59) + 2*(60-89) + 3*(90+)
  3. IncomePerDependent       : MonthlyIncome / (NumberOfDependents + 1)
  4. EstimatedMonthlyDebt     : MonthlyIncome * DebtRatio
  5. DisposableIncome         : MonthlyIncome - EstimatedMonthlyDebt
  6. UnsecuredLinesRatio      : (OpenLoans - RealEstateLoans) / (OpenLoans + 1)
  7. MissingIncomeFlag        : 1 if MonthlyIncome was missing else 0
  8. OverUtilizationFlag      : 1 if RevolvingUtilization > 1.0 else 0
=============================================================================
"""

import pandas as pd
import numpy as np
import os
import sys

# Ensure src/ is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import FEATURES, TARGET, RANDOM_STATE, TEST_SIZE
from data_preprocessing import load_data, clean_data, split_data


ENGINEERED_FEATURE_NAMES = [
    "TotalPastDueCount",
    "DelinquencySeverityIndex",
    "IncomePerDependent",
    "EstimatedMonthlyDebt",
    "DisposableIncome",
    "UnsecuredLinesRatio",
    "MissingIncomeFlag",
    "OverUtilizationFlag",
]

ALL_ENGINEERED_FEATURES = FEATURES + ENGINEERED_FEATURE_NAMES


def engineer_features(df: pd.DataFrame, income_median: float = None) -> tuple[pd.DataFrame, float]:
    """
    Construct 8 domain-specific financial features on the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing the cleaned 10 features.
    income_median : float, optional
        Precomputed training set median for MonthlyIncome.
        If None, computes median from df (MUST only be done on training set).

    Returns
    -------
    df_eng : pd.DataFrame
        Dataframe with 18 features (+ TARGET if present).
    income_median : float
        The median used for imputation of income interactions.
    """
    df = df.copy()

    # Track missing income before imputation
    missing_income_flag = df["MonthlyIncome"].isnull().astype(float)

    # Median calculation / application
    if income_median is None:
        income_median = float(df["MonthlyIncome"].median())

    # Temporary series for calculations
    imputed_income = df["MonthlyIncome"].fillna(income_median)
    imputed_dependents = df["NumberOfDependents"].fillna(0.0)

    # 1. Total Past Due Count
    df["TotalPastDueCount"] = (
        df["NumberOfTime30-59DaysPastDueNotWorse"].fillna(0)
        + df["NumberOfTime60-89DaysPastDueNotWorse"].fillna(0)
        + df["NumberOfTimes90DaysLate"].fillna(0)
    )

    # 2. Delinquency Severity Index
    df["DelinquencySeverityIndex"] = (
        1.0 * df["NumberOfTime30-59DaysPastDueNotWorse"].fillna(0)
        + 2.0 * df["NumberOfTime60-89DaysPastDueNotWorse"].fillna(0)
        + 3.0 * df["NumberOfTimes90DaysLate"].fillna(0)
    )

    # 3. Income Per Dependent
    df["IncomePerDependent"] = imputed_income / (imputed_dependents + 1.0)

    # 4. Estimated Monthly Debt ($) - Capped to avoid extreme multiplication with outlier ratios
    # In GMSC, DebtRatio for zero/missing income is often large; we clamp DebtRatio to 99th percentile for multiplication
    debt_ratio_clamped = df["DebtRatio"].clip(upper=10.0)
    df["EstimatedMonthlyDebt"] = imputed_income * debt_ratio_clamped

    # 5. Disposable Income ($)
    df["DisposableIncome"] = imputed_income - df["EstimatedMonthlyDebt"]

    # 6. Unsecured Credit Lines Ratio
    open_loans = df["NumberOfOpenCreditLinesAndLoans"].fillna(0)
    re_loans = df["NumberRealEstateLoansOrLines"].fillna(0)
    unsecured = np.maximum(0, open_loans - re_loans)
    df["UnsecuredLinesRatio"] = unsecured / (open_loans + 1.0)

    # 7. Missing Income Flag
    df["MissingIncomeFlag"] = missing_income_flag

    # 8. Over-Utilization Flag (> 100% credit limit)
    util = df["RevolvingUtilizationOfUnsecuredLines"].fillna(0)
    df["OverUtilizationFlag"] = (util > 1.0).astype(float)

    # Impute remaining missing values in original columns for clean model ingestion
    df["MonthlyIncome"] = imputed_income
    df["NumberOfDependents"] = imputed_dependents
    if "age" in df.columns:
        df["age"] = df["age"].fillna(df["age"].median())

    return df, income_median


def prepare_engineered_splits():
    """
    Load, clean, split, and engineer features with zero data leakage.

    Returns
    -------
    X_train_eng, X_test_eng, y_train, y_test
    """
    df_raw = load_data()
    df_clean, _ = clean_data(df_raw, verbose=False)
    X_train, X_test, y_train, y_test = split_data(df_clean)

    print(f"\n[Feature Engineering] Engineering features for Train (N={len(X_train):,})...")
    X_train_eng, train_income_median = engineer_features(X_train, income_median=None)

    print(f"[Feature Engineering] Transforming Test using Train Median (${train_income_median:,.2f})...")
    X_test_eng, _ = engineer_features(X_test, income_median=train_income_median)

    # Ensure correct columns and no target leakage
    feature_cols = [c for c in ALL_ENGINEERED_FEATURES if c in X_train_eng.columns]
    X_train_eng = X_train_eng[feature_cols]
    X_test_eng = X_test_eng[feature_cols]

    print(f"[Feature Engineering] Success! Engineered Shape: {X_train_eng.shape}")
    print(f"Features: {list(X_train_eng.columns)}")

    return X_train_eng, X_test_eng, y_train, y_test


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te = prepare_engineered_splits()
    print("Null check train:\n", X_tr.isnull().sum())
    print("Null check test:\n", X_te.isnull().sum())
