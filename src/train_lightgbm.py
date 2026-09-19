"""
=============================================================================
MODEL 5 -- LIGHTGBM CLASSIFIER TRAINING
=============================================================================
Project : GMSC Credit Scoring Research
File    : src/train_lightgbm.py
Purpose : Train LightGBM with stratified 10-fold CV.
          GridSearchCV over learning_rate, num_leaves, and n_estimators.
          Evaluate best model on untouched 30% test set.

Why LightGBM?
  - Fast, memory-efficient histogram-based gradient boosting (Ke et al., NeurIPS 2017)
  - Grows trees leaf-wise (best-first) rather than level-wise (depth-wise)
  - Often achieves state-of-the-art accuracy on tabular financial datasets
  - Directly compares against XGBoost in our ensemble benchmarking

Pipeline Structure (leakage-safe):
  [Imputer + Scaler fitted strictly per fold] --> [LGBMClassifier]

Outputs:
  - models/lightgbm.joblib
  - results/lightgbm_metrics.csv
=============================================================================
RUN COMMAND:
  cd credit-scoring-project
  python src/train_lightgbm.py
=============================================================================
"""

import sys, os
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.metrics         import make_scorer, f1_score, recall_score, precision_score
from lightgbm                import LGBMClassifier

from config             import RANDOM_STATE, CV_FOLDS, MODEL_DIR, RESULT_DIR
from data_preprocessing import load_data, clean_data, split_data
from models             import build_lightgbm_pipeline
from evaluation         import evaluate_model, print_cv_results, save_results_table


LGBM_PARAM_GRID = {
    "classifier__learning_rate": [0.03, 0.05, 0.1],
    "classifier__num_leaves":    [15, 31, 63],
    "classifier__n_estimators":  [100, 200],
}


def f1_class1(y_true, y_pred):
    return f1_score(y_true, y_pred, pos_label=1, zero_division=0)

def recall_class1(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=1, zero_division=0)

def precision_class1(y_true, y_pred):
    return precision_score(y_true, y_pred, pos_label=1, zero_division=0)


def train_lightgbm():
    print("=" * 65)
    print("MODEL 5 -- LIGHTGBM CLASSIFIER TRAINING")
    print("=" * 65)
    print(f"\n  Model               : LightGBM (Light Gradient Boosting Machine)")
    print(f"  learning_rate grid  : {LGBM_PARAM_GRID['classifier__learning_rate']}")
    print(f"  num_leaves grid     : {LGBM_PARAM_GRID['classifier__num_leaves']}")
    print(f"  n_estimators grid   : {LGBM_PARAM_GRID['classifier__n_estimators']}")
    print(f"  CV strategy         : Stratified {CV_FOLDS}-fold (training data only)")
    print(f"  Scoring             : ROC-AUC")

    n_combinations = (
        len(LGBM_PARAM_GRID["classifier__learning_rate"])
        * len(LGBM_PARAM_GRID["classifier__num_leaves"])
        * len(LGBM_PARAM_GRID["classifier__n_estimators"])
    )
    print(f"  Total CV fits       : {n_combinations} x {CV_FOLDS} = {n_combinations * CV_FOLDS}")
    print(f"  Random state        : {RANDOM_STATE}")

    # 1. Load data
    print("\n[Step 1] Load -> Clean -> Split")
    df_raw      = load_data()
    df_clean, _ = clean_data(df_raw, verbose=False)
    X_train, X_test, y_train, y_test = split_data(df_clean)

    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = neg_count / pos_count

    print(f"  Training set        : {X_train.shape} | Class 1: {pos_count} ({y_train.mean()*100:.2f}%)")
    print(f"  Test set            : {X_test.shape}  | Class 1: {int(y_test.sum())}  ({y_test.mean()*100:.2f}%)")
    print(f"  scale_pos_weight    : {scale_pos_weight:.2f}")

    # 2. Stratified CV
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # 3. Grid Search
    print(f"\n[Step 2] GridSearchCV across {n_combinations} combinations ({n_combinations * CV_FOLDS} fits)...")
    base_pipeline = build_lightgbm_pipeline(scale_pos_weight=scale_pos_weight)

    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=LGBM_PARAM_GRID,
        cv=skf,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    grid_search.fit(X_train, y_train)

    # 4. Results
    print(f"\n[Step 3] GridSearchCV results (Top 5)")
    cv_results_df = pd.DataFrame(grid_search.cv_results_).sort_values("mean_test_score", ascending=False)
    print(f"\n  {'lr':>6}  {'leaves':>8}  {'n_est':>6}  {'Mean AUC':>10}  {'Std AUC':>10}")
    print(f"  {'-'*48}")
    for i, (_, row) in enumerate(cv_results_df.head(5).iterrows()):
        lr   = row["param_classifier__learning_rate"]
        nl   = row["param_classifier__num_leaves"]
        ne   = row["param_classifier__n_estimators"]
        mean = row["mean_test_score"]
        std  = row["std_test_score"]
        marker = " <-- BEST" if i == 0 else ""
        print(f"  {lr:>6}  {nl:>8}  {ne:>6}  {mean:>10.4f}  {std:>10.4f}{marker}")

    bp = grid_search.best_params_
    print(f"\n  Best learning_rate : {bp['classifier__learning_rate']}")
    print(f"  Best num_leaves    : {bp['classifier__num_leaves']}")
    print(f"  Best n_estimators  : {bp['classifier__n_estimators']}")
    print(f"  Best CV ROC-AUC    : {grid_search.best_score_:.4f}")

    # 5. Full CV scoring
    print(f"\n[Step 4] Full CV scoring with best parameters")
    best_pipeline = build_lightgbm_pipeline(
        learning_rate=bp["classifier__learning_rate"],
        num_leaves=bp["classifier__num_leaves"],
        n_estimators=bp["classifier__n_estimators"],
        scale_pos_weight=scale_pos_weight,
    )
    scoring = {
        "roc_auc": "roc_auc",
        "f1_class1": make_scorer(f1_class1),
        "recall_c1": make_scorer(recall_class1),
        "precision_c1": make_scorer(precision_class1),
        "accuracy": "accuracy",
    }
    cv_scores = cross_validate(best_pipeline, X_train, y_train, cv=skf, scoring=scoring, n_jobs=-1)
    print_cv_results({
        "ROC-AUC": cv_scores["test_roc_auc"],
        "F1 (class 1)": cv_scores["test_f1_class1"],
        "Recall (class 1)": cv_scores["test_recall_c1"],
        "Precision (class 1)": cv_scores["test_precision_c1"],
        "Accuracy": cv_scores["test_accuracy"],
    }, model_name="LightGBM (10-fold CV)")

    # 6. Final test set evaluation
    print(f"\n[Step 5] Evaluating on untouched 30% test set")
    final_model = grid_search.best_estimator_
    y_pred = final_model.predict(X_test)
    y_prob = final_model.predict_proba(X_test)[:, 1]

    results = evaluate_model(
        model_name="LightGBM",
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        verbose=True,
    )

    # 7. Save model and results
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "lightgbm.joblib")
    joblib.dump(final_model, model_path)
    print(f"\n[Step 6] Model saved: {model_path}")

    os.makedirs(RESULT_DIR, exist_ok=True)
    results_path = os.path.join(RESULT_DIR, "lightgbm_metrics.csv")
    save_results_table([results], results_path)

    print(f"\n{'='*65}")
    print(f"LIGHTGBM COMPLETE -- Test ROC-AUC: {results['ROC_AUC']:.4f} | Recall: {results['Recall_Class1']:.4f}")
    print(f"{'='*65}")
    return final_model, results


if __name__ == "__main__":
    train_lightgbm()
