"""
=============================================================================
DATA PREPROCESSING
=============================================================================
Project : GMSC Credit Scoring Research
File    : src/data_preprocessing.py
Purpose : Load, clean, split, and build leakage-safe preprocessing pipelines.

Cleaning order (per master specification):
  Step 1 -- Load and drop index column
  Step 2 -- Convert age == 0 to NaN (biologically invalid)
  Step 3 -- Cap 96/98 artifact values using 99th percentile of valid values
  (Split happens in Phase 4 -- split_data())
  (Imputation + scaling fitted on train only -- Phase 5 -- build_pipeline())

Reference: Vakrani et al. (2026) -- adapted to Give Me Some Credit dataset
=============================================================================
"""

import pandas as pd
import numpy as np
import os
import sys

# Make sure src/ is importable when running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DATA_PATH, RANDOM_STATE, TEST_SIZE, TARGET, FEATURES,
    ARTIFACT_COLS, ARTIFACT_VALUES, ARTIFACT_CAP_QUANTILE,
    MEDIAN_IMPUTE_COLS, MODE_IMPUTE_COLS,
)


# =============================================================================
# STEP 1 -- LOAD
# =============================================================================

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load cs-training.csv and drop the unnamed leading index column.

    Returns
    -------
    pd.DataFrame
        Raw data with index column removed. No other changes.
    """
    print(f"[load_data] Loading: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    print(f"[load_data] Raw shape: {df.shape}")

    # Drop unnamed leading index column
    unnamed_cols = [c for c in df.columns if "unnamed" in c.lower()]
    first_col    = df.columns[0]
    is_index_like = (
        df[first_col].is_monotonic_increasing
        and df[first_col].nunique() == len(df)
        and pd.api.types.is_integer_dtype(df[first_col])
    )

    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
        print(f"[load_data] Dropped unnamed columns: {unnamed_cols}")
    elif is_index_like:
        df = df.drop(columns=[first_col])
        print(f"[load_data] Dropped index-like column: '{first_col}'")

    print(f"[load_data] Shape after index drop: {df.shape}")
    return df


# =============================================================================
# STEP 2 + 3 -- CLEAN
# =============================================================================

def clean_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Apply known data-quality fixes to the raw DataFrame.

    Fixes applied (no imputation -- that is fitted on training data only):
      1. age == 0  -->  NaN   (biologically impossible value)
      2. 96/98 artifacts in delinquency columns  -->  capped at
         99th percentile of valid (non-96/98) values in that column.

    Parameters
    ----------
    df      : raw DataFrame (output of load_data)
    verbose : print detailed report

    Returns
    -------
    pd.DataFrame  (cleaned copy -- original not modified)
    """
    df = df.copy()
    print("\n[clean_data] Starting data cleaning...")

    # ------------------------------------------------------------------
    # Fix 1 -- Invalid age (age == 0 -> NaN)
    # ------------------------------------------------------------------
    age_zero_mask = df["age"] == 0
    n_age_zero    = age_zero_mask.sum()

    if n_age_zero > 0:
        df.loc[age_zero_mask, "age"] = np.nan
        if verbose:
            print(f"\n  [age] {n_age_zero} row(s) with age==0 converted to NaN.")
            print(f"  [age] These will be median-imputed from TRAINING DATA only in Phase 5.")
    else:
        print("\n  [age] No age==0 values found.")

    # ------------------------------------------------------------------
    # Fix 2 -- 96/98 artifact capping
    # ------------------------------------------------------------------
    print(f"\n  [artifacts] Capping 96/98 values at {int(ARTIFACT_CAP_QUANTILE*100)}th "
          f"percentile of valid observations.")
    print(f"  [artifacts] 'Valid' means values not in {ARTIFACT_VALUES}.")
    print(f"  [artifacts] Columns affected: {ARTIFACT_COLS}")

    cap_values = {}

    for col in ARTIFACT_COLS:
        # Isolate valid values (exclude 96 and 98)
        valid_mask   = ~df[col].isin(ARTIFACT_VALUES)
        valid_values = df.loc[valid_mask, col]

        # Calculate cap from valid observations
        cap = valid_values.quantile(ARTIFACT_CAP_QUANTILE)
        cap_values[col] = cap

        # Count rows that will be changed
        artifact_mask = df[col].isin(ARTIFACT_VALUES)
        n_artifacts   = artifact_mask.sum()

        # Apply cap
        df.loc[artifact_mask, col] = cap

        if verbose:
            print(f"\n    Column : {col}")
            print(f"    Valid obs used for cap : {valid_mask.sum():,}")
            print(f"    99th percentile cap    : {cap:.4f}")
            print(f"    Artifact rows capped   : {n_artifacts:,}")
            print(f"    Values replaced with   : {cap:.4f}")

    # ------------------------------------------------------------------
    # Verification report
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n[clean_data] Post-cleaning verification:")
        print(f"  age NaN count      : {df['age'].isnull().sum()}")
        for col in ARTIFACT_COLS:
            remaining = df[col].isin(ARTIFACT_VALUES).sum()
            print(f"  {col} artifacts remaining : {remaining}")

        print(f"\n[clean_data] Shape unchanged: {df.shape}")
        print(f"[clean_data] Cleaning complete. Dataset NOT saved to disk.")
        print(f"             (Save will happen only with explicit permission.)")

    return df, cap_values


# =============================================================================
# STEP 4 -- SPLIT (Phase 4)
# =============================================================================

def split_data(df: pd.DataFrame):
    """
    Stratified 70/30 train/test split.

    Parameters
    ----------
    df : cleaned DataFrame (output of clean_data)

    Returns
    -------
    X_train, X_test, y_train, y_test  (all as DataFrames / Series)
    """
    from sklearn.model_selection import train_test_split

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    print(f"\n[split_data] Splitting dataset...")
    print(f"  Total rows : {len(df):,}")
    print(f"  Test size  : {TEST_SIZE} (stratified, random_state={RANDOM_STATE})")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print(f"\n  X_train : {X_train.shape}  |  y_train class 1 rate: "
          f"{y_train.mean()*100:.4f}%")
    print(f"  X_test  : {X_test.shape}   |  y_test  class 1 rate: "
          f"{y_test.mean()*100:.4f}%")

    return X_train, X_test, y_train, y_test


# =============================================================================
# STEP 5 -- LEAKAGE-SAFE PREPROCESSING PIPELINE (Phase 5)
# =============================================================================

def build_preprocessing_pipeline():
    """
    Build a scikit-learn Pipeline for leakage-safe imputation + scaling.

    Rules:
      - Median imputation for : MonthlyIncome, age
      - Mode  imputation for  : NumberOfDependents
      - StandardScaler applied to all features
      - Pipeline must be fit on TRAINING DATA ONLY
      - Same fitted pipeline transforms test data

    Returns
    -------
    sklearn Pipeline
    """
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer

    median_cols = MEDIAN_IMPUTE_COLS          # ["MonthlyIncome", "age"]
    mode_cols   = MODE_IMPUTE_COLS            # ["NumberOfDependents"]
    other_cols  = [
        f for f in FEATURES
        if f not in median_cols and f not in mode_cols
    ]

    median_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    mode_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    passthrough_pipe = Pipeline([
        ("passthrough", SimpleImputer(strategy="median")),  # safety fallback
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("median_impute", median_pipe,       median_cols),
            ("mode_impute",   mode_pipe,         mode_cols),
            ("other",         passthrough_pipe,  other_cols),
        ],
        remainder="drop",
    )

    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("scaler",        StandardScaler()),
    ])

    print("[build_preprocessing_pipeline] Pipeline constructed.")
    print(f"  Median imputed : {median_cols}")
    print(f"  Mode imputed   : {mode_cols}")
    print(f"  Other cols     : {other_cols}")
    print(f"  Final step     : StandardScaler")
    print(f"  IMPORTANT      : Call pipeline.fit() on TRAINING DATA ONLY.")

    return full_pipeline


# =============================================================================
# QUICK VERIFICATION (run directly)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 3 -- DATA CLEANING VERIFICATION")
    print("=" * 60)

    df_raw           = load_data()
    df_clean, caps   = clean_data(df_raw, verbose=True)

    print("\n" + "=" * 60)
    print("Cap values applied to 96/98 artifacts:")
    for col, cap in caps.items():
        print(f"  {col}: cap = {cap:.4f}")

    print("\nMissing values after cleaning:")
    print(df_clean[FEATURES].isnull().sum().to_string())

    print("\nPhase 3 complete. No files written. No split performed.")
    print("=" * 60)
