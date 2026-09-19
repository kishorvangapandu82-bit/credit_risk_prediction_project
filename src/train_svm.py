"""
=============================================================================
MODEL 6 -- SUPPORT VECTOR MACHINE (SVM) TRAINING
=============================================================================
Project : GMSC Credit Scoring Research
File    : src/train_svm.py
Purpose : Train Linear Support Vector Machine (LinearSVC with Calibration)
          using stratified 10-fold CV.
          GridSearchCV over regularization parameter C.
          Evaluate best model on untouched 30% test set.

Why SVM?
  - Finds the optimal maximum-margin hyperplane separating classes
  - Effective in high-dimensional feature spaces
  - Provides a geometric / maximum-margin linear baseline contrasting with
    probabilistic Logistic Regression, tree-based models, and neural nets
  - Uses CalibratedClassifierCV to output well-calibrated probabilities for ROC-AUC

Pipeline Structure (leakage-safe):
  [Imputer + Scaler fitted strictly per fold] --> [Calibrated LinearSVC]

Outputs:
  - models/svm.joblib
  - results/svm_metrics.csv
=============================================================================
RUN COMMAND:
  cd credit-scoring-project
  python src/train_svm.py
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
from sklearn.svm             import LinearSVC
from sklearn.calibration     import CalibratedClassifierCV

from config             import RANDOM_STATE, CV_FOLDS, MODEL_DIR, RESULT_DIR
from data_preprocessing import load_data, clean_data, split_data, build_preprocessing_pipeline
from models             import build_svm_pipeline
from evaluation         import evaluate_model, print_cv_results, save_results_table


SVM_PARAM_GRID = {
    "classifier__estimator__C": [0.01, 0.1, 1.0, 10.0],
}


def f1_class1(y_true, y_pred):
    return f1_score(y_true, y_pred, pos_label=1, zero_division=0)

def recall_class1(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=1, zero_division=0)

def precision_class1(y_true, y_pred):
    return precision_score(y_true, y_pred, pos_label=1, zero_division=0)


def train_svm():
    print("=" * 65)
    print("MODEL 6 -- SUPPORT VECTOR MACHINE (SVM) TRAINING")
    print("=" * 65)
    print(f"\n  Model               : Support Vector Machine (Linear SVM with Calibration)")
    print(f"  C grid              : {SVM_PARAM_GRID['classifier__estimator__C']}")
    print(f"  CV strategy         : Stratified {CV_FOLDS}-fold (training data only)")
    print(f"  Scoring             : ROC-AUC")

    n_combinations = len(SVM_PARAM_GRID["classifier__estimator__C"])
    print(f"  Total CV fits       : {n_combinations} x {CV_FOLDS} = {n_combinations * CV_FOLDS}")
    print(f"  Random state        : {RANDOM_STATE}")

    # 1. Load data
    print("\n[Step 1] Load -> Clean -> Split")
    df_raw      = load_data()
    df_clean, _ = clean_data(df_raw, verbose=False)
    X_train, X_test, y_train, y_test = split_data(df_clean)

    # 2. Stratified CV
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # 3. Grid Search
    print(f"\n[Step 2] GridSearchCV across {n_combinations} C values ({n_combinations * CV_FOLDS} fits)...")
    base_pipeline = build_svm_pipeline()

    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=SVM_PARAM_GRID,
        cv=skf,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    grid_search.fit(X_train, y_train)

    # 4. Results
    print(f"\n[Step 3] GridSearchCV results")
    cv_results_df = pd.DataFrame(grid_search.cv_results_).sort_values("mean_test_score", ascending=False)
    print(f"\n  {'C':>8}  {'Mean AUC':>10}  {'Std AUC':>10}")
    print(f"  {'-'*35}")
    for i, (_, row) in enumerate(cv_results_df.iterrows()):
        c_val = row["param_classifier__estimator__C"]
        mean  = row["mean_test_score"]
        std   = row["std_test_score"]
        marker = " <-- BEST" if i == 0 else ""
        print(f"  {c_val:>8}  {mean:>10.4f}  {std:>10.4f}{marker}")

    bp = grid_search.best_params_
    best_c = bp["classifier__estimator__C"]
    print(f"\n  Best C          : {best_c}")
    print(f"  Best CV ROC-AUC : {grid_search.best_score_:.4f}")

    # 5. Full CV scoring
    print(f"\n[Step 4] Full CV scoring with best C parameter")
    best_pipeline = build_svm_pipeline(C=best_c)
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
    }, model_name="Support Vector Machine (10-fold CV)")

    # 6. Final test set evaluation
    print(f"\n[Step 5] Evaluating on untouched 30% test set")
    final_model = grid_search.best_estimator_
    y_pred = final_model.predict(X_test)
    y_prob = final_model.predict_proba(X_test)[:, 1]

    results = evaluate_model(
        model_name="Support Vector Machine",
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        verbose=True,
    )

    # 7. Save model and results
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "svm.joblib")
    joblib.dump(final_model, model_path)
    print(f"\n[Step 6] Model saved: {model_path}")

    os.makedirs(RESULT_DIR, exist_ok=True)
    results_path = os.path.join(RESULT_DIR, "svm_metrics.csv")
    save_results_table([results], results_path)

    print(f"\n{'='*65}")
    print(f"SVM COMPLETE -- Test ROC-AUC: {results['ROC_AUC']:.4f} | Recall: {results['Recall_Class1']:.4f}")
    print(f"{'='*65}")
    return final_model, results


if __name__ == "__main__":
    train_svm()
