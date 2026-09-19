"""
=============================================================================
CENTRAL CONFIGURATION
=============================================================================
Project : GMSC Credit Scoring Research
File    : src/config.py
Purpose : Single source of truth for all project parameters.
          Import from here; do NOT duplicate constants in other files.

Reference: Vakrani et al. (2026) -- adapted to Give Me Some Credit dataset
=============================================================================
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data",    "cs-training.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")
PLOT_DIR   = os.path.join(BASE_DIR, "plots")

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42          # Fixed seed for all random operations

# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
TEST_SIZE = 0.30           # 70% train / 30% test (stratified)

# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
CV_FOLDS = 10              # Stratified 10-fold CV on training data only

# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------
TARGET = "SeriousDlqin2yrs"
# 0 = No serious delinquency within 2 years (majority class)
# 1 = Serious delinquency / default (minority / BAD outcome)

# ---------------------------------------------------------------------------
# Features (exactly 10 -- do not add or remove without permission)
# ---------------------------------------------------------------------------
FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

# ---------------------------------------------------------------------------
# Imputation columns
# ---------------------------------------------------------------------------
MEDIAN_IMPUTE_COLS = ["MonthlyIncome", "age"]   # median imputation
MODE_IMPUTE_COLS   = ["NumberOfDependents"]      # mode imputation

# ---------------------------------------------------------------------------
# 96/98 artifact columns
# ---------------------------------------------------------------------------
ARTIFACT_COLS   = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
]
ARTIFACT_VALUES = [96, 98]
ARTIFACT_CAP_QUANTILE = 0.99   # 99th percentile of valid values used as cap

# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------
MLP_EPOCHS     = 50
MLP_HIDDEN_1   = 64
MLP_HIDDEN_2   = 32
MLP_ACTIVATION = "relu"

# ---------------------------------------------------------------------------
# AUC warning threshold (requires leakage investigation if exceeded)
# ---------------------------------------------------------------------------
AUC_LEAKAGE_THRESHOLD = 0.90

# ---------------------------------------------------------------------------
# Stress test scenario
# ---------------------------------------------------------------------------
STRESS_INCOME_FACTOR   = 0.80   # MonthlyIncome * 0.80
STRESS_DEBTRATIO_FACTOR = 1.25  # DebtRatio * 1.25  (derived from income drop)
