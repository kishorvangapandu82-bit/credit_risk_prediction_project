"""
=============================================================================
PHASE 8 -- XGBOOST CLASSIFIER TRAINING
=============================================================================
Project : GMSC Credit Scoring Research
File    : src/train_xgboost.py
Purpose : Train XGBoost with stratified 10-fold CV.
          GridSearchCV over learning_rate, max_depth, and n_estimators.
          Evaluate best model on untouched 30% test set.

Why XGBoost?
  - Extreme Gradient Boosting (Chen & Guestrin, 2016)
  - Ensembles multiple shallow trees sequentially, each correcting residual errors
  - Built-in regularization (L1/L2) prevents overfitting better than single trees
  - Handles non-linear feature interactions and threshold effects automatically
  - Handles class imbalance via scale_pos_weight = count(negative) / count(positive)
  - Dominant state-of-the-art benchmark in tabular credit risk modeling

Hyperparameters to Tune:
  - learning_rate  : Step size shrinkage [0.01, 0.05, 0.1, 0.2]
  - max_depth      : Maximum depth of a tree [3, 4, 5, 6]
  - n_estimators   : Number of boosting rounds [100, 200, 300]
  - scale_pos_weight: Fixed to train set ratio (~13.96) to balance positive class

Pipeline Structure (leakage-safe):
  [Imputer + Scaler fitted strictly per fold] --> [XGBClassifier]

=============================================================================
RUN COMMAND:
  cd credit-scoring-project
  python src/train_xgboost.py
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
from sklearn.metrics          import make_scorer, f1_score, recall_score, precision_score
from xgboost                  import XGBClassifier

from config               import RANDOM_STATE, CV_FOLDS, MODEL_DIR, RESULT_DIR
from data_preprocessing   import load_data, clean_data, split_data
from models               import build_xgboost_pipeline
from evaluation           import evaluate_model, print_cv_results, save_results_table


# ---------------------------------------------------------------------------
# Hyperparameter search space (XGBoost -- per methodology)
# ---------------------------------------------------------------------------
XGB_PARAM_GRID = {
    "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "classifier__max_depth":     [3, 4, 5, 6],
    "classifier__n_estimators":  [100, 200, 300],
}


def f1_class1(y_true, y_pred):
    return f1_score(y_true, y_pred, pos_label=1, zero_division=0)

def recall_class1(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=1, zero_division=0)

def precision_class1(y_true, y_pred):
    return precision_score(y_true, y_pred, pos_label=1, zero_division=0)


def train_xgboost():

    print("=" * 65)
    print("PHASE 8 -- XGBOOST CLASSIFIER TRAINING")
    print("=" * 65)
    print(f"\n  Model               : XGBoost (Extreme Gradient Boosting)")
    print(f"  learning_rate grid  : {XGB_PARAM_GRID['classifier__learning_rate']}")
    print(f"  max_depth grid      : {XGB_PARAM_GRID['classifier__max_depth']}")
    print(f"  n_estimators grid   : {XGB_PARAM_GRID['classifier__n_estimators']}")
    print(f"  CV strategy         : Stratified {CV_FOLDS}-fold (training data only)")
    print(f"  Scoring             : ROC-AUC")
    
    n_combinations = (
        len(XGB_PARAM_GRID["classifier__learning_rate"])
        * len(XGB_PARAM_GRID["classifier__max_depth"])
        * len(XGB_PARAM_GRID["classifier__n_estimators"])
    )
    print(f"  Total CV fits       : {n_combinations} x {CV_FOLDS} = {n_combinations * CV_FOLDS}")
    print(f"  Random state        : {RANDOM_STATE}")

    # ------------------------------------------------------------------
    # 1. Load -> Clean -> Split
    # ------------------------------------------------------------------
    print("\n[Step 1] Load -> Clean -> Split")
    df_raw         = load_data()
    df_clean, _    = clean_data(df_raw, verbose=False)
    X_train, X_test, y_train, y_test = split_data(df_clean)

    # Compute scale_pos_weight strictly on training split
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = neg_count / pos_count

    print(f"  Training set        : {X_train.shape} | Class 1: {pos_count} ({y_train.mean()*100:.2f}%)")
    print(f"  Test set            : {X_test.shape}  | Class 1: {int(y_test.sum())}  ({y_test.mean()*100:.2f}%)")
    print(f"  scale_pos_weight    : {scale_pos_weight:.2f} (count(0) / count(1) on training set)")

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
    print(f"\n[Step 2] GridSearchCV -- {n_combinations * CV_FOLDS} total fits ...")
    print(f"         (Tuning across {n_combinations} parameter combinations)")

    base_pipeline = build_xgboost_pipeline(scale_pos_weight=scale_pos_weight)

    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=XGB_PARAM_GRID,
        cv=skf,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    grid_search.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 4. CV results table
    # ------------------------------------------------------------------
    print(f"\n[Step 3] GridSearchCV results (sorted by mean AUC)")
    cv_results_df = pd.DataFrame(grid_search.cv_results_)
    cv_results_df = cv_results_df.sort_values("mean_test_score", ascending=False)

    print(f"\n  {'lr':>6}  {'depth':>6}  {'n_est':>6}  {'Mean AUC':>10}  {'Std AUC':>10}")
    print(f"  {'-'*46}")
    for i, (_, row) in enumerate(cv_results_df.head(10).iterrows()):
        lr     = row["param_classifier__learning_rate"]
        d      = row["param_classifier__max_depth"]
        ne     = row["param_classifier__n_estimators"]
        mean   = row["mean_test_score"]
        std    = row["std_test_score"]
        marker = " <-- BEST" if i == 0 else ""
        print(f"  {lr:>6}  {d:>6}  {ne:>6}  {mean:>10.4f}  {std:>10.4f}{marker}")

    bp = grid_search.best_params_
    print(f"\n  Best learning_rate : {bp['classifier__learning_rate']}")
    print(f"  Best max_depth     : {bp['classifier__max_depth']}")
    print(f"  Best n_estimators  : {bp['classifier__n_estimators']}")
    print(f"  Best CV ROC-AUC    : {grid_search.best_score_:.4f}")

    # ------------------------------------------------------------------
    # 5. Full CV scoring with best config
    # ------------------------------------------------------------------
    print(f"\n[Step 4] Full CV scoring with best parameters")
    best_pipeline = build_xgboost_pipeline(
        learning_rate=bp["classifier__learning_rate"],
        max_depth=bp["classifier__max_depth"],
        n_estimators=bp["classifier__n_estimators"],
        scale_pos_weight=scale_pos_weight,
    )

    scoring = {
        "roc_auc"     : "roc_auc",
        "f1_class1"   : make_scorer(f1_class1),
        "recall_c1"   : make_scorer(recall_class1),
        "precision_c1": make_scorer(precision_class1),
        "accuracy"    : "accuracy",
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
        "ROC-AUC"            : cv_scores["test_roc_auc"],
        "F1 (class 1)"       : cv_scores["test_f1_class1"],
        "Recall (class 1)"   : cv_scores["test_recall_c1"],
        "Precision (class 1)": cv_scores["test_precision_c1"],
        "Accuracy"           : cv_scores["test_accuracy"],
    }, model_name="XGBoost (10-fold CV)")

    # ------------------------------------------------------------------
    # 6. Final model (refitted by GridSearchCV on full training set)
    # ------------------------------------------------------------------
    print(f"\n[Step 5] Final model (refitted on full training set)")
    final_model = grid_search.best_estimator_

    # ------------------------------------------------------------------
    # 7. Model structure & feature importances
    # ------------------------------------------------------------------
    xgb_clf = final_model.named_steps["classifier"]
    from config import FEATURES
    importances = xgb_clf.feature_importances_
    feat_imp = sorted(zip(FEATURES, importances), key=lambda x: x[1], reverse=True)
    
    print(f"\n  Feature importances (Gain-based):")
    for feat, imp in feat_imp:
        bar = "#" * int(imp * 50)
        print(f"    {feat:<45} {imp:.4f}  {bar}")

    # ------------------------------------------------------------------
    # 8. Evaluate on untouched test set
    # ------------------------------------------------------------------
    print(f"\n[Step 6] Evaluating on untouched 30% test set")
    y_pred = final_model.predict(X_test)
    y_prob = final_model.predict_proba(X_test)[:, 1]

    results = evaluate_model(
        model_name="XGBoost",
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        verbose=True,
    )

    # ------------------------------------------------------------------
    # 9. Save model
    # ------------------------------------------------------------------
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "xgboost.joblib")
    joblib.dump(final_model, model_path)
    print(f"\n[Step 7] Model saved to: {model_path}")

    # ------------------------------------------------------------------
    # 10. Save results
    # ------------------------------------------------------------------
    os.makedirs(RESULT_DIR, exist_ok=True)
    results_path = os.path.join(RESULT_DIR, "xgboost_metrics.csv")
    save_results_table([results], results_path)

    # ------------------------------------------------------------------
    # 11. Multi-Model Comparison Table
    # ------------------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"MODEL COMPARISON SO FAR (Untouched 30% Test Set)")
    print(f"{'='*65}")
    print(f"  {'Model':<22} {'ROC-AUC':>10} {'Recall':>10} {'Precision':>10} {'F1':>10} {'Accuracy':>10}")
    print(f"  {'-'*65}")

    all_models = []
    lr_file = os.path.join(RESULT_DIR, "logistic_regression_metrics.csv")
    dt_file = os.path.join(RESULT_DIR, "decision_tree_metrics.csv")

    if os.path.exists(lr_file):
        lr_df = pd.read_csv(lr_file)
        all_models.append(lr_df.iloc[0])
    if os.path.exists(dt_file):
        dt_df = pd.read_csv(dt_file)
        all_models.append(dt_df.iloc[0])
    all_models.append(pd.Series(results))

    for m in all_models:
        print(f"  {m['Model']:<22} {m['ROC_AUC']:>10.4f} {m['Recall_Class1']:>10.4f} {m['Precision_Class1']:>10.4f} {m['F1_Class1']:>10.4f} {m['Accuracy']:>10.4f}")

    # ------------------------------------------------------------------
    # 12. Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"PHASE 8 COMPLETE -- XGBOOST")
    print(f"{'='*65}")
    print(f"  Best learning_rate    : {bp['classifier__learning_rate']}")
    print(f"  Best max_depth        : {bp['classifier__max_depth']}")
    print(f"  Best n_estimators     : {bp['classifier__n_estimators']}")
    print(f"  CV ROC-AUC (mean)     : {cv_scores['test_roc_auc'].mean():.4f}")
    print(f"  Test ROC-AUC          : {results['ROC_AUC']:.4f}")
    print(f"  Test Precision (c1)   : {results['Precision_Class1']:.4f}")
    print(f"  Test Recall (c1)      : {results['Recall_Class1']:.4f}")
    print(f"  Test F1 (c1)          : {results['F1_Class1']:.4f}")
    print(f"  Test Accuracy         : {results['Accuracy']:.4f}")
    print(f"  Model saved           : {model_path}")
    print(f"{'='*65}")

    return final_model, results


if __name__ == "__main__":
    train_xgboost()
