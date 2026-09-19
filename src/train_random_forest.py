"""
=============================================================================
MODEL 7 -- RANDOM FOREST CLASSIFIER TRAINING
=============================================================================
Project : GMSC Credit Scoring Research
File    : src/train_random_forest.py
Purpose : Train Random Forest with stratified 10-fold CV.
          GridSearchCV over n_estimators, max_depth, and min_samples_split.
          Evaluate best model on untouched 30% test set.

Why Random Forest?
  - Bagging ensemble of decorrelated decision trees (Breiman, 2001)
  - Each tree grown on a bootstrap sample; only sqrt(p) features considered
    at each split -- reduces variance vs. a single CART tree without
    sacrificing interpretable feature importances
  - Built-in out-of-bag (OOB) error estimate as a free validation signal
  - class_weight='balanced' scales loss by inverse class frequency,
    addressing the 6.68% minority-class imbalance without SMOTE
  - Complementary to gradient boosting (XGBoost/LightGBM): uses
    parallel bagging instead of sequential residual correction

Hyperparameters to Tune:
  - n_estimators      : Number of trees [100, 200, 300]
  - max_depth         : Maximum depth per tree [None, 10, 20]  (None=fully grown)
  - min_samples_split : Minimum samples required to split a node [10, 20, 50]

Pipeline Structure (leakage-safe):
  [Imputer + Scaler fitted strictly per fold] --> [RandomForestClassifier]

Outputs:
  - models/random_forest.joblib
  - results/random_forest_metrics.csv
=============================================================================
RUN COMMAND:
  cd credit-scoring-project
  python src/train_random_forest.py
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

from config             import RANDOM_STATE, CV_FOLDS, MODEL_DIR, RESULT_DIR, FEATURES
from data_preprocessing import load_data, clean_data, split_data
from models             import build_random_forest_pipeline
from evaluation         import evaluate_model, print_cv_results, save_results_table


# ---------------------------------------------------------------------------
# Hyperparameter search space
# ---------------------------------------------------------------------------
RF_PARAM_GRID = {
    "classifier__n_estimators":     [100, 200, 300],
    "classifier__max_depth":        [None, 10, 20],
    "classifier__min_samples_split":[10, 20, 50],
}


def f1_class1(y_true, y_pred):
    return f1_score(y_true, y_pred, pos_label=1, zero_division=0)

def recall_class1(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=1, zero_division=0)

def precision_class1(y_true, y_pred):
    return precision_score(y_true, y_pred, pos_label=1, zero_division=0)


def train_random_forest():

    print("=" * 65)
    print("MODEL 7 -- RANDOM FOREST CLASSIFIER TRAINING")
    print("=" * 65)
    print(f"\n  Model                  : Random Forest (Bagging Ensemble)")
    print(f"  n_estimators grid      : {RF_PARAM_GRID['classifier__n_estimators']}")
    print(f"  max_depth grid         : {RF_PARAM_GRID['classifier__max_depth']}")
    print(f"  min_samples_split grid : {RF_PARAM_GRID['classifier__min_samples_split']}")
    print(f"  CV strategy            : Stratified {CV_FOLDS}-fold (training data only)")
    print(f"  Scoring                : ROC-AUC")
    print(f"  Class balancing        : class_weight='balanced'")

    n_combinations = (
        len(RF_PARAM_GRID["classifier__n_estimators"])
        * len(RF_PARAM_GRID["classifier__max_depth"])
        * len(RF_PARAM_GRID["classifier__min_samples_split"])
    )
    print(f"  Total CV fits          : {n_combinations} x {CV_FOLDS} = {n_combinations * CV_FOLDS}")
    print(f"  Random state           : {RANDOM_STATE}")

    # ------------------------------------------------------------------
    # 1. Load -> Clean -> Split
    # ------------------------------------------------------------------
    print("\n[Step 1] Load -> Clean -> Split")
    df_raw         = load_data()
    df_clean, _    = clean_data(df_raw, verbose=False)
    X_train, X_test, y_train, y_test = split_data(df_clean)

    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())

    print(f"  Training set : {X_train.shape} | Class 1: {pos_count} ({y_train.mean()*100:.2f}%)")
    print(f"  Test set     : {X_test.shape}  | Class 1: {int(y_test.sum())}  ({y_test.mean()*100:.2f}%)")
    print(f"  Imbalance ratio (neg/pos) : {neg_count/pos_count:.2f}  --> handled via class_weight='balanced'")

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

    base_pipeline = build_random_forest_pipeline()

    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=RF_PARAM_GRID,
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

    print(f"\n  {'n_est':>6}  {'depth':>8}  {'min_split':>10}  {'Mean AUC':>10}  {'Std AUC':>10}")
    print(f"  {'-'*54}")
    for i, (_, row) in enumerate(cv_results_df.head(10).iterrows()):
        ne  = row["param_classifier__n_estimators"]
        d   = row["param_classifier__max_depth"]
        ms  = row["param_classifier__min_samples_split"]
        mean = row["mean_test_score"]
        std  = row["std_test_score"]
        d_str = str(d) if d is not None else "None"
        marker = " <-- BEST" if i == 0 else ""
        print(f"  {ne:>6}  {d_str:>8}  {ms:>10}  {mean:>10.4f}  {std:>10.4f}{marker}")

    bp = grid_search.best_params_
    print(f"\n  Best n_estimators     : {bp['classifier__n_estimators']}")
    print(f"  Best max_depth        : {bp['classifier__max_depth']}")
    print(f"  Best min_samples_split: {bp['classifier__min_samples_split']}")
    print(f"  Best CV ROC-AUC       : {grid_search.best_score_:.4f}")

    # ------------------------------------------------------------------
    # 5. Full CV scoring with best config
    # ------------------------------------------------------------------
    print(f"\n[Step 4] Full CV scoring with best parameters")
    best_pipeline = build_random_forest_pipeline(
        n_estimators=bp["classifier__n_estimators"],
        max_depth=bp["classifier__max_depth"],
        min_samples_split=bp["classifier__min_samples_split"],
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
    }, model_name="Random Forest (10-fold CV)")

    # ------------------------------------------------------------------
    # 6. Final model (refitted by GridSearchCV on full training set)
    # ------------------------------------------------------------------
    print(f"\n[Step 5] Final model (refitted on full training set)")
    final_model = grid_search.best_estimator_

    # ------------------------------------------------------------------
    # 7. Feature importances (Gini / Mean Decrease in Impurity)
    # ------------------------------------------------------------------
    rf_clf = final_model.named_steps["classifier"]
    importances = rf_clf.feature_importances_
    feat_imp = sorted(zip(FEATURES, importances), key=lambda x: x[1], reverse=True)

    print(f"\n  Feature importances (Mean Decrease in Impurity):")
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
        model_name="Random Forest",
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        verbose=True,
    )

    # ------------------------------------------------------------------
    # 9. Save model
    # ------------------------------------------------------------------
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "random_forest.joblib")
    joblib.dump(final_model, model_path)
    print(f"\n[Step 7] Model saved to: {model_path}")

    # ------------------------------------------------------------------
    # 10. Save results
    # ------------------------------------------------------------------
    os.makedirs(RESULT_DIR, exist_ok=True)
    results_path = os.path.join(RESULT_DIR, "random_forest_metrics.csv")
    save_results_table([results], results_path)

    # ------------------------------------------------------------------
    # 11. Multi-Model Comparison Table
    # ------------------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"MODEL COMPARISON SO FAR (Untouched 30% Test Set)")
    print(f"{'='*65}")
    print(f"  {'Model':<22} {'ROC-AUC':>10} {'Recall':>10} {'Precision':>10} {'F1':>10} {'Accuracy':>10}")
    print(f"  {'-'*65}")

    all_model_files = [
        ("logistic_regression_metrics.csv",),
        ("decision_tree_metrics.csv",),
        ("mlp_metrics.csv",),
        ("xgboost_metrics.csv",),
        ("lightgbm_metrics.csv",),
        ("svm_metrics.csv",),
        ("random_forest_metrics.csv",),
    ]
    for (fname,) in all_model_files:
        fpath = os.path.join(RESULT_DIR, fname)
        if os.path.exists(fpath):
            m = pd.read_csv(fpath).iloc[0]
            print(f"  {m['Model']:<22} {m['ROC_AUC']:>10.4f} {m['Recall_Class1']:>10.4f} "
                  f"{m['Precision_Class1']:>10.4f} {m['F1_Class1']:>10.4f} {m['Accuracy']:>10.4f}")

    # ------------------------------------------------------------------
    # 12. Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"MODEL 7 COMPLETE -- RANDOM FOREST")
    print(f"{'='*65}")
    print(f"  Best n_estimators      : {bp['classifier__n_estimators']}")
    print(f"  Best max_depth         : {bp['classifier__max_depth']}")
    print(f"  Best min_samples_split : {bp['classifier__min_samples_split']}")
    print(f"  CV ROC-AUC (mean)      : {cv_scores['test_roc_auc'].mean():.4f}")
    print(f"  Test ROC-AUC           : {results['ROC_AUC']:.4f}")
    print(f"  Test Precision (c1)    : {results['Precision_Class1']:.4f}")
    print(f"  Test Recall (c1)       : {results['Recall_Class1']:.4f}")
    print(f"  Test F1 (c1)           : {results['F1_Class1']:.4f}")
    print(f"  Test Accuracy          : {results['Accuracy']:.4f}")
    print(f"  Model saved            : {model_path}")
    print(f"{'='*65}")

    return final_model, results


if __name__ == "__main__":
    train_random_forest()
