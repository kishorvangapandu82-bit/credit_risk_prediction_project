"""
=============================================================================
EVALUATION METRICS
=============================================================================
Project : GMSC Credit Scoring Research
File    : src/evaluation.py
Purpose : Compute and report all evaluation metrics for every model.
          All metrics refer to class 1 (serious delinquency = BAD outcome).

Metrics:
  Accuracy      -- overall (reported but NOT the primary metric)
  Precision     -- class 1 only
  Recall        -- class 1 only
  F1            -- class 1 only
  ROC-AUC       -- discriminative ability (primary metric)

AUC Warning: If ROC-AUC >= 0.90, a leakage investigation is triggered.

Reference: Vakrani et al. (2026) -- adapted to Give Me Some Credit dataset
=============================================================================
"""

import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from config import AUC_LEAKAGE_THRESHOLD


def evaluate_model(
    model_name: str,
    y_true,
    y_pred,
    y_prob,
    verbose: bool = True,
) -> dict:
    """
    Compute all required metrics for one model on one dataset.

    Parameters
    ----------
    model_name : str    -- label for display
    y_true     : array  -- true labels (0 or 1)
    y_pred     : array  -- predicted class labels
    y_prob     : array  -- predicted probability for class 1
    verbose    : bool   -- print report

    Returns
    -------
    dict with keys: Model, Accuracy, Precision_Class1,
                    Recall_Class1, F1_Class1, ROC_AUC
    """
    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall    = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1        = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    roc_auc   = roc_auc_score(y_true, y_prob)

    results = {
        "Model":            model_name,
        "Accuracy":         round(accuracy,  4),
        "Precision_Class1": round(precision, 4),
        "Recall_Class1":    round(recall,    4),
        "F1_Class1":        round(f1,        4),
        "ROC_AUC":          round(roc_auc,   4),
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"EVALUATION REPORT: {model_name}")
        print(f"{'='*60}")
        print(f"  Accuracy              : {accuracy:.4f}")
        print(f"  Precision (class 1)   : {precision:.4f}")
        print(f"  Recall    (class 1)   : {recall:.4f}")
        print(f"  F1 Score  (class 1)   : {f1:.4f}")
        print(f"  ROC-AUC               : {roc_auc:.4f}")
        print(f"\n  Note: Accuracy is NOT the primary metric.")
        print(f"        Class 1 (serious delinquency) is only ~6.7% of data.")
        print(f"        Primary metrics: ROC-AUC, Recall, F1 (class 1).")

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print(f"\n  Confusion Matrix:")
        print(f"    {'':>20}  Pred 0    Pred 1")
        print(f"    {'Actual 0':>20}  {cm[0,0]:>6}    {cm[0,1]:>6}")
        print(f"    {'Actual 1':>20}  {cm[1,0]:>6}    {cm[1,1]:>6}")

        # AUC leakage warning
        if roc_auc >= AUC_LEAKAGE_THRESHOLD:
            print(f"\n  {'!'*60}")
            print(f"  WARNING: ROC-AUC >= {AUC_LEAKAGE_THRESHOLD}")
            print(f"  This result requires leakage investigation")
            print(f"  before being interpreted as genuine model performance.")
            print(f"\n  Investigate:")
            print(f"    1. Was imputation fitted before splitting?")
            print(f"    2. Was scaling fitted before splitting?")
            print(f"    3. Was SMOTE applied to test data?")
            print(f"    4. Were duplicate records shared across train/test?")
            print(f"    5. Was the target accidentally included as a feature?")
            print(f"    6. Was any post-outcome information used?")
            print(f"    7. Was the test set accidentally used during tuning?")
            print(f"  {'!'*60}")

    return results


def print_cv_results(cv_scores: dict, model_name: str):
    """
    Print cross-validation fold scores summary.

    Parameters
    ----------
    cv_scores  : dict of {metric_name: array_of_fold_scores}
    model_name : str
    """
    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION RESULTS: {model_name}")
    print(f"{'='*60}")
    for metric, scores in cv_scores.items():
        print(f"  {metric:<25} : {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
    print(f"{'='*60}")


def save_results_table(results_list: list, output_path: str):
    """
    Save final metrics to results/final_metrics.csv.

    Parameters
    ----------
    results_list : list of dicts (one per model)
    output_path  : absolute path to output CSV
    """
    df = pd.DataFrame(results_list, columns=[
        "Model", "Accuracy", "Precision_Class1",
        "Recall_Class1", "F1_Class1", "ROC_AUC"
    ])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n[save_results_table] Saved to: {output_path}")
    print(df.to_string(index=False))
    return df
