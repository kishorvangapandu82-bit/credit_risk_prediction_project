"""
=============================================================================
PHASE 2 -- DATASET INSPECTION (READ-ONLY)
=============================================================================
Project : GMSC Credit Scoring Research
File    : src/inspect_data.py
Purpose : Inspect cs-training.csv and report dataset facts.
          Does NOT modify the dataset in any way.

Reference: Vakrani et al. (2026) -- adapted to Give Me Some Credit dataset
=============================================================================
"""

import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "cs-training.csv")

# ---------------------------------------------------------------------------
# Features we will use (defined here for inspection cross-check only)
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
TARGET = "SeriousDlqin2yrs"

ARTIFACT_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
]
ARTIFACT_VALUES = [96, 98]


def separator(title=""):
    width = 70
    if title:
        pad = (width - len(title) - 2) // 2
        print("\n" + "=" * pad + f" {title} " + "=" * pad)
    else:
        print("\n" + "=" * width)


def inspect_data():
    # -----------------------------------------------------------------------
    # 1. Load
    # -----------------------------------------------------------------------
    separator("LOADING DATASET")
    print(f"Path : {DATA_PATH}")

    if not os.path.exists(DATA_PATH):
        print("\n[ERROR] cs-training.csv not found at the expected path.")
        print("Please ensure the file is at: data/cs-training.csv")
        return

    df_raw = pd.read_csv(DATA_PATH)
    print(f"Raw shape (before any cleaning) : {df_raw.shape}")
    print(f"\nAll column names ({len(df_raw.columns)}):")
    for i, col in enumerate(df_raw.columns):
        print(f"  [{i:02d}] {col}")

    # -----------------------------------------------------------------------
    # 2. Identify and drop unnamed leading index column
    # -----------------------------------------------------------------------
    separator("INDEX COLUMN DETECTION")
    unnamed_cols = [c for c in df_raw.columns if "unnamed" in c.lower() or c.strip() == ""]
    first_col = df_raw.columns[0]
    is_index_like = (
        df_raw[first_col].is_monotonic_increasing
        and df_raw[first_col].nunique() == len(df_raw)
        and pd.api.types.is_integer_dtype(df_raw[first_col])
    )

    print(f"First column        : '{first_col}'")
    print(f"Unnamed columns     : {unnamed_cols}")
    print(f"First col is index  : {is_index_like}")

    if unnamed_cols:
        df = df_raw.drop(columns=unnamed_cols)
        print(f"Dropped unnamed columns: {unnamed_cols}")
    elif is_index_like:
        df = df_raw.drop(columns=[first_col])
        print(f"Dropped index-like column: '{first_col}'")
    else:
        df = df_raw.copy()
        print("No index column detected to drop.")

    print(f"\nShape after dropping index : {df.shape}")

    # -----------------------------------------------------------------------
    # 3. Basic structure
    # -----------------------------------------------------------------------
    separator("BASIC STRUCTURE")
    print(f"Rows    : {df.shape[0]:,}")
    print(f"Columns : {df.shape[1]}")
    print(f"\nColumn names present : {list(df.columns)}")

    print(f"\n--- Required feature check ---")
    missing_features = [f for f in FEATURES if f not in df.columns]
    present_features = [f for f in FEATURES if f in df.columns]
    print(f"Required features found     : {len(present_features)} / {len(FEATURES)}")
    if missing_features:
        print(f"[WARNING] Missing features  : {missing_features}")
    else:
        print("All 10 required features present.")

    target_present = TARGET in df.columns
    print(f"Target '{TARGET}' present   : {target_present}")

    # -----------------------------------------------------------------------
    # 4. Data types
    # -----------------------------------------------------------------------
    separator("DATA TYPES")
    print(df.dtypes.to_string())

    # -----------------------------------------------------------------------
    # 5. Missing values
    # -----------------------------------------------------------------------
    separator("MISSING VALUES")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(4)
    missing_df = pd.DataFrame({
        "Missing_Count": missing,
        "Missing_Pct(%)": missing_pct
    })
    missing_with_vals = missing_df[missing_df["Missing_Count"] > 0]

    if missing_with_vals.empty:
        print("No missing values detected.")
    else:
        print(missing_with_vals.to_string())

    print(f"\nTotal missing cells : {missing.sum():,}")

    # -----------------------------------------------------------------------
    # 6. Target distribution
    # -----------------------------------------------------------------------
    separator("TARGET DISTRIBUTION")
    target_counts = df[TARGET].value_counts().sort_index()
    target_pct = (df[TARGET].value_counts(normalize=True) * 100).sort_index().round(4)

    print(f"Class 0 (No serious delinquency) : {target_counts.get(0, 0):>8,}  ({target_pct.get(0, 0):.4f}%)")
    print(f"Class 1 (Serious delinquency)    : {target_counts.get(1, 0):>8,}  ({target_pct.get(1, 0):.4f}%)")
    print(f"Total                            : {len(df):>8,}")

    actual_positive_rate = target_pct.get(1, 0)
    print(f"\nActual positive-class rate       : {actual_positive_rate:.4f}%")
    if abs(actual_positive_rate - 6.7) > 1.0:
        print(f"[NOTE] Positive rate differs from expected ~6.7% -- verify.")
    else:
        print(f"Positive rate consistent with expected ~6.7%.")

    # -----------------------------------------------------------------------
    # 7. Duplicate rows
    # -----------------------------------------------------------------------
    separator("DUPLICATE ROWS")
    n_duplicates = df.duplicated().sum()
    print(f"Exact duplicate rows (all columns) : {n_duplicates:,}")
    n_feature_dups = df[FEATURES].duplicated().sum()
    print(f"Duplicate feature rows (10 cols)   : {n_feature_dups:,}")

    # -----------------------------------------------------------------------
    # 8. Invalid age values
    # -----------------------------------------------------------------------
    separator("INVALID AGE VALUES")
    age_zero = (df["age"] == 0).sum()
    age_negative = (df["age"] < 0).sum()
    age_extreme = (df["age"] > 100).sum()
    print(f"Age == 0 (invalid -- will become NaN)  : {age_zero:,}")
    print(f"Age < 0  (negative -- invalid)          : {age_negative:,}")
    print(f"Age > 100 (extreme)                    : {age_extreme:,}")
    print(f"\nAge statistics:")
    print(df["age"].describe().to_string())

    # -----------------------------------------------------------------------
    # 9. 96/98 artifact check
    # -----------------------------------------------------------------------
    separator("96/98 DATA-ENTRY ARTIFACTS")
    for col in ARTIFACT_COLS:
        for val in ARTIFACT_VALUES:
            count = (df[col] == val).sum()
            pct = count / len(df) * 100
            print(f"  {col:<45} == {val} : {count:>6,}  ({pct:.4f}%)")

    print(f"\nTotal artifact rows across all 3 columns:")
    total_artifacts = sum(
        (df[col] == val).sum()
        for col in ARTIFACT_COLS
        for val in ARTIFACT_VALUES
    )
    mask = df[ARTIFACT_COLS].isin(ARTIFACT_VALUES).any(axis=1)
    rows_with_artifacts = mask.sum()
    print(f"  Total artifact entries (cells) : {total_artifacts:,}")
    print(f"  Rows containing an artifact    : {rows_with_artifacts:,}")

    # -----------------------------------------------------------------------
    # 10. Quick feature statistics
    # -----------------------------------------------------------------------
    separator("FEATURE SUMMARY STATISTICS")
    print(df[FEATURES].describe().T.to_string())

    # -----------------------------------------------------------------------
    # 11. Final summary report
    # -----------------------------------------------------------------------
    separator("PHASE 2 -- SUMMARY REPORT")
    print(f"""
  Rows                          : {df.shape[0]:,}
  Columns (after index drop)    : {df.shape[1]}
  Required features present     : {len(present_features)} / 10
  Target present                : {target_present}

  Missing -- MonthlyIncome      : {df["MonthlyIncome"].isnull().sum():,}  ({df["MonthlyIncome"].isnull().mean()*100:.2f}%)
  Missing -- NumberOfDependents : {df["NumberOfDependents"].isnull().sum():,}  ({df["NumberOfDependents"].isnull().mean()*100:.2f}%)
  Missing -- age                : {df["age"].isnull().sum():,}
  Missing -- other features     : {df[FEATURES].drop(columns=["MonthlyIncome","NumberOfDependents","age"]).isnull().sum().sum():,}

  Class 0 count                 : {target_counts.get(0, 0):,}
  Class 1 count                 : {target_counts.get(1, 0):,}
  Positive rate                 : {actual_positive_rate:.4f}%

  Duplicate rows (full)         : {n_duplicates:,}
  Invalid age (==0)             : {age_zero:,}
  96/98 artifact rows           : {rows_with_artifacts:,}
""")
    separator()
    print("Phase 2 inspection complete. Dataset NOT modified.")
    separator()


if __name__ == "__main__":
    inspect_data()
