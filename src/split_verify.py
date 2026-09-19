"""
=============================================================================
PHASE 4 -- TRAIN/TEST SPLIT VERIFICATION
=============================================================================
Project : GMSC Credit Scoring Research
File    : src/split_verify.py
Purpose : Run the 70/30 stratified split and verify correctness.
          Does NOT impute, scale, or train any model.
          Does NOT save the split to disk (that happens in train.py).

Pipeline verified here:
  load_data() -> clean_data() -> split_data()
=============================================================================
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RANDOM_STATE, TEST_SIZE, CV_FOLDS, FEATURES, TARGET
from data_preprocessing import load_data, clean_data, split_data


def verify_split():

    print("=" * 65)
    print("PHASE 4 -- TRAIN/TEST SPLIT VERIFICATION")
    print("=" * 65)
    print(f"  Random state : {RANDOM_STATE}")
    print(f"  Test size    : {TEST_SIZE} (30%)")
    print(f"  Train size   : {1 - TEST_SIZE} (70%)")
    print(f"  Stratified   : Yes (on target column)")

    # ------------------------------------------------------------------
    # Step 1: Load
    # ------------------------------------------------------------------
    print("\n--- Step 1: Load ---")
    df_raw = load_data()

    # ------------------------------------------------------------------
    # Step 2: Clean
    # ------------------------------------------------------------------
    print("\n--- Step 2: Clean ---")
    df_clean, cap_values = clean_data(df_raw, verbose=False)
    print(f"  Shape after cleaning : {df_clean.shape}")

    # ------------------------------------------------------------------
    # Step 3: Split
    # ------------------------------------------------------------------
    print("\n--- Step 3: Split ---")
    X_train, X_test, y_train, y_test = split_data(df_clean)

    # ------------------------------------------------------------------
    # Verification Report
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("SPLIT VERIFICATION REPORT")
    print("=" * 65)

    total = len(X_train) + len(X_test)

    print(f"\n  Full dataset rows        : {total:,}")
    print(f"\n  TRAINING SET")
    print(f"    Rows                   : {len(X_train):,}")
    print(f"    % of total             : {len(X_train)/total*100:.2f}%")
    print(f"    Class 0 count          : {(y_train==0).sum():,}")
    print(f"    Class 1 count          : {(y_train==1).sum():,}")
    print(f"    Class 1 rate           : {y_train.mean()*100:.4f}%")

    print(f"\n  TEST SET  (untouched from here on)")
    print(f"    Rows                   : {len(X_test):,}")
    print(f"    % of total             : {len(X_test)/total*100:.2f}%")
    print(f"    Class 0 count          : {(y_test==0).sum():,}")
    print(f"    Class 1 count          : {(y_test==1).sum():,}")
    print(f"    Class 1 rate           : {y_test.mean()*100:.4f}%")

    # Stratification check
    print(f"\n  STRATIFICATION CHECK")
    full_rate  = (y_train.tolist() + y_test.tolist()).count(1) / total * 100
    train_rate = y_train.mean() * 100
    test_rate  = y_test.mean()  * 100
    diff       = abs(train_rate - test_rate)

    print(f"    Full dataset class 1 rate  : {full_rate:.4f}%")
    print(f"    Train set   class 1 rate   : {train_rate:.4f}%")
    print(f"    Test set    class 1 rate   : {test_rate:.4f}%")
    print(f"    Difference (train vs test) : {diff:.6f}%")

    if diff < 0.1:
        print(f"    Result : STRATIFICATION SUCCESSFUL (diff < 0.1%)")
    else:
        print(f"    Result : [WARNING] Larger than expected difference.")

    # Missing values in each split (pre-imputation)
    print(f"\n  MISSING VALUES IN TRAINING SET (pre-imputation)")
    train_missing = X_train.isnull().sum()
    train_missing = train_missing[train_missing > 0]
    for col, cnt in train_missing.items():
        print(f"    {col:<45} : {cnt:,}  ({cnt/len(X_train)*100:.2f}%)")

    print(f"\n  MISSING VALUES IN TEST SET (pre-imputation)")
    test_missing = X_test.isnull().sum()
    test_missing = test_missing[test_missing > 0]
    for col, cnt in test_missing.items():
        print(f"    {col:<45} : {cnt:,}  ({cnt/len(X_test)*100:.2f}%)")

    # Cross-validation note
    print(f"\n  CROSS-VALIDATION PLAN (Phase 6 onward)")
    print(f"    Strategy : Stratified {CV_FOLDS}-fold CV")
    print(f"    Data     : Training set ONLY ({len(X_train):,} rows)")
    print(f"    Purpose  : Model selection & hyperparameter tuning")
    print(f"    Test set : Remains untouched until final evaluation")

    print("\n" + "=" * 65)
    print("Phase 4 complete. No data saved. No imputation performed.")
    print("No models trained. Test set status: UNTOUCHED.")
    print("=" * 65)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    verify_split()
