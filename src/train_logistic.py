"""
=============================================================================
PHASE 6 -- LOGISTIC REGRESSION TRAINING
=============================================================================
Project : GMSC Credit Scoring Research
File    : src/train_logistic.py
Purpose : Train Logistic Regression with stratified 10-fold CV.
          Select best C via CV. Train final model on full training set.
          Evaluate on untouched 30% test set.

Why Logistic Regression?
  - Traditional statistical baseline used in credit scoring since the 1970s
  - Provides interpretable coefficients
  - Requires standardized features (handled by pipeline)
  - class_weight="balanced" compensates for 6.68% minority class
  - Serves as the benchmark all ML models must beat

Why class_weight="balanced"?
  - Without it, the model predicts class 0 for almost every sample
  - "balanced" scales the loss contribution of each class inversely
    to its frequency: rare class 1 gets higher weight per sample
  - This improves Recall for class 1 (catching actual defaults)

C parameter (regularization):
  - C = 1/lambda (inverse of regularization strength)
  - Small C = stronger regularization (simpler model, may underfit)
  - Large C = weaker regularization (complex model, may overfit)
  - We search: [0.001, 0.01, 0.1, 1.0, 10.0] via 10-fold CV

Pipeline structure (leakage-safe):
  [Imputer + Scaler fitted per fold] --> [LogisticRegression]

=============================================================================
RUN COMMAND:
  cd credit-scoring-project
  python src/train_logistic.py
=============================================================================
"""

import sys, os
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.model_selection  import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.metrics          import make_scorer, roc_auc_score

from config               import RANDOM_STATE, CV_FOLDS, MODEL_DIR, RESULT_DIR
from data_preprocessing   import load_data, clean_data, split_data
from models               import build_logistic_pipeline
from evaluation           import evaluate_model, print_cv_results, save_results_table


# ---------------------------------------------------------------------------
# Hyperparameter search space (Logistic Regression)
# ---------------------------------------------------------------------------
LR_PARAM_GRID = {
    "classifier__C": [0.001, 0.01, 0.1, 1.0, 10.0],
}


def train_logistic_regression():

    print("=" * 65)
    print("PHASE 6 -- LOGISTIC REGRESSION TRAINING")
    print("=" * 65)
    print(f"\n  Model        : Logistic Regression")
    print(f"  class_weight : balanced")
    print(f"  C grid       : {LR_PARAM_GRID['classifier__C']}")
    print(f"  CV strategy  : Stratified {CV_FOLDS}-fold (on training data only)")
    print(f"  Scoring      : ROC-AUC")
    print(f"  Random state : {RANDOM_STATE}")

    # ------------------------------------------------------------------
    # 1. Load -> Clean -> Split
    # ------------------------------------------------------------------
    print("\n[Step 1] Load -> Clean -> Split")
    df_raw         = load_data()
    df_clean, _    = clean_data(df_raw, verbose=False)
    X_train, X_test, y_train, y_test = split_data(df_clean)

    print(f"  Training set : {X_train.shape} | Class 1: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
    print(f"  Test set     : {X_test.shape}  | Class 1: {y_test.sum()}  ({y_test.mean()*100:.2f}%)")

    # ------------------------------------------------------------------
    # 2. Stratified K-Fold setup
    # ------------------------------------------------------------------
    skf = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # ------------------------------------------------------------------
    # 3. GridSearchCV -- hyperparameter tuning (training data only)
    # ------------------------------------------------------------------
    print(f"\n[Step 2] GridSearchCV over C values: {LR_PARAM_GRID['classifier__C']}")
    print(f"         Using {CV_FOLDS}-fold stratified CV on training data...")
    print(f"         Total fits: {len(LR_PARAM_GRID['classifier__C'])} x {CV_FOLDS} = "
          f"{len(LR_PARAM_GRID['classifier__C']) * CV_FOLDS}")
    print(f"         (This may take 1-3 minutes)")

    base_pipeline = build_logistic_pipeline()

    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=LR_PARAM_GRID,
        cv=skf,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    grid_search.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 4. CV results
    # ------------------------------------------------------------------
    print(f"\n[Step 3] GridSearchCV results")
    cv_results_df = pd.DataFrame(grid_search.cv_results_)
    print(f"\n  {'C value':>10}  {'Mean AUC':>10}  {'Std AUC':>10}")
    print(f"  {'-'*35}")
    for _, row in cv_results_df.iterrows():
        c_val = row["param_classifier__C"]
        mean  = row["mean_test_score"]
        std   = row["std_test_score"]
        marker = " <-- BEST" if c_val == grid_search.best_params_["classifier__C"] else ""
        print(f"  {c_val:>10}  {mean:>10.4f}  {std:>10.4f}{marker}")

    print(f"\n  Best C        : {grid_search.best_params_['classifier__C']}")
    print(f"  Best CV AUC   : {grid_search.best_score_:.4f}")

    # ------------------------------------------------------------------
    # 5. Full CV metrics with best model (additional scoring metrics)
    # ------------------------------------------------------------------
    print(f"\n[Step 4] Full CV scoring with best C on training data")
    best_pipeline = build_logistic_pipeline(C=grid_search.best_params_["classifier__C"])

    scoring = {
        "roc_auc"   : "roc_auc",
        "f1_class1" : make_scorer(f1_score_class1),
        "recall_c1" : make_scorer(recall_score_class1),
        "precision_c1": make_scorer(precision_score_class1),
        "accuracy"  : "accuracy",
    }

    cv_scores = cross_validate(
        best_pipeline,
        X_train, y_train,
        cv=skf,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )

    print_cv_results({
        "ROC-AUC"           : cv_scores["test_roc_auc"],
        "F1 (class 1)"      : cv_scores["test_f1_class1"],
        "Recall (class 1)"  : cv_scores["test_recall_c1"],
        "Precision (class 1)": cv_scores["test_precision_c1"],
        "Accuracy"          : cv_scores["test_accuracy"],
    }, model_name="Logistic Regression (10-fold CV)")

    # ------------------------------------------------------------------
    # 6. Final model -- already refitted by GridSearchCV (refit=True)
    # ------------------------------------------------------------------
    print(f"\n[Step 5] Final model (fitted on full training set by GridSearchCV refit)")
    final_model = grid_search.best_estimator_

    # ------------------------------------------------------------------
    # 7. Evaluate on untouched test set
    # ------------------------------------------------------------------
    print(f"\n[Step 6] Evaluating on untouched 30% test set")
    y_pred = final_model.predict(X_test)
    y_prob = final_model.predict_proba(X_test)[:, 1]

    results = evaluate_model(
        model_name="Logistic Regression",
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        verbose=True,
    )

    # ------------------------------------------------------------------
    # 8. Save model
    # ------------------------------------------------------------------
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "logistic_regression.joblib")
    joblib.dump(final_model, model_path)
    print(f"\n[Step 7] Model saved to: {model_path}")

    # ------------------------------------------------------------------
    # 9. Save results
    # ------------------------------------------------------------------
    os.makedirs(RESULT_DIR, exist_ok=True)
    results_path = os.path.join(RESULT_DIR, "logistic_regression_metrics.csv")
    save_results_table([results], results_path)

    # ------------------------------------------------------------------
    # 10. Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"PHASE 6 COMPLETE -- LOGISTIC REGRESSION")
    print(f"{'='*65}")
    print(f"  Best C              : {grid_search.best_params_['classifier__C']}")
    print(f"  CV ROC-AUC (mean)   : {cv_scores['test_roc_auc'].mean():.4f}")
    print(f"  Test ROC-AUC        : {results['ROC_AUC']}")
    print(f"  Test Precision (c1) : {results['Precision_Class1']}")
    print(f"  Test Recall (c1)    : {results['Recall_Class1']}")
    print(f"  Test F1 (c1)        : {results['F1_Class1']}")
    print(f"  Test Accuracy       : {results['Accuracy']}")
    print(f"  Model saved         : {model_path}")
    print(f"{'='*65}")

    return final_model, results


# ---------------------------------------------------------------------------
# Helper scorers for cross_validate
# ---------------------------------------------------------------------------
from sklearn.metrics import f1_score, recall_score, precision_score

def f1_score_class1(y_true, y_pred):
    return f1_score(y_true, y_pred, pos_label=1, zero_division=0)

def recall_score_class1(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=1, zero_division=0)

def precision_score_class1(y_true, y_pred):
    return precision_score(y_true, y_pred, pos_label=1, zero_division=0)


if __name__ == "__main__":
    train_logistic_regression()
