"""
=============================================================================
PHASE 7 -- DECISION TREE (CART) TRAINING
=============================================================================
Project : GMSC Credit Scoring Research
File    : src/train_decision_tree.py
Purpose : Train Decision Tree (CART) with stratified 10-fold CV.
          GridSearchCV over max_depth and min_samples_split.
          Evaluate best model on untouched 30% test set.

Why Decision Tree?
  - Non-parametric model -- no assumption about feature distributions
  - Naturally handles non-linear relationships
  - Highly interpretable (tree structure is human-readable)
  - Contrast to Logistic Regression: no scaling requirement,
    but scaling still applied (harmless and consistent)
  - Representative of shallow rule-based ML approaches in credit scoring

Gini impurity (criterion):
  - Measures probability that a randomly chosen sample is misclassified
  - Gini = 1 - sum(p_i^2) for each class i
  - Lower Gini = purer node = better split
  - Fixed at "gini" per methodology specification

max_depth:
  - Controls tree complexity. Too deep = overfitting. Too shallow = underfitting.
  - Search: [3, 4, 5, 6, 7, 8, 10]

min_samples_split:
  - Minimum samples required to split an internal node
  - Higher values = more conservative splits = simpler tree
  - Search: [5, 10, 20, 50]

Pipeline structure (leakage-safe):
  [Imputer + Scaler fitted per fold] --> [DecisionTreeClassifier]

=============================================================================
RUN COMMAND:
  cd credit-scoring-project
  python src/train_decision_tree.py
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
from sklearn.tree             import DecisionTreeClassifier

from config               import RANDOM_STATE, CV_FOLDS, MODEL_DIR, RESULT_DIR
from data_preprocessing   import load_data, clean_data, split_data
from models               import build_decision_tree_pipeline
from evaluation           import evaluate_model, print_cv_results, save_results_table


# ---------------------------------------------------------------------------
# Hyperparameter search space (Decision Tree -- per methodology)
# ---------------------------------------------------------------------------
DT_PARAM_GRID = {
    "classifier__max_depth":         [3, 4, 5, 6, 7, 8, 10],
    "classifier__min_samples_split": [5, 10, 20, 50],
    "classifier__criterion":         ["gini"],
}


def f1_class1(y_true, y_pred):
    return f1_score(y_true, y_pred, pos_label=1, zero_division=0)

def recall_class1(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=1, zero_division=0)

def precision_class1(y_true, y_pred):
    return precision_score(y_true, y_pred, pos_label=1, zero_division=0)


def train_decision_tree():

    print("=" * 65)
    print("PHASE 7 -- DECISION TREE (CART) TRAINING")
    print("=" * 65)
    print(f"\n  Model            : Decision Tree (CART)")
    print(f"  criterion        : gini (fixed)")
    print(f"  class_weight     : balanced")
    print(f"  max_depth grid   : {DT_PARAM_GRID['classifier__max_depth']}")
    print(f"  min_split grid   : {DT_PARAM_GRID['classifier__min_samples_split']}")
    print(f"  CV strategy      : Stratified {CV_FOLDS}-fold (training data only)")
    print(f"  Scoring          : ROC-AUC")
    print(f"  Total CV fits    : "
          f"{len(DT_PARAM_GRID['classifier__max_depth'])} x "
          f"{len(DT_PARAM_GRID['classifier__min_samples_split'])} x "
          f"{CV_FOLDS} = "
          f"{len(DT_PARAM_GRID['classifier__max_depth']) * len(DT_PARAM_GRID['classifier__min_samples_split']) * CV_FOLDS}")
    print(f"  Random state     : {RANDOM_STATE}")

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
    n_fits = (len(DT_PARAM_GRID["classifier__max_depth"])
              * len(DT_PARAM_GRID["classifier__min_samples_split"])
              * CV_FOLDS)
    print(f"\n[Step 2] GridSearchCV -- {n_fits} total fits ...")
    print(f"         (This may take 2-5 minutes)")

    base_pipeline = build_decision_tree_pipeline()

    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=DT_PARAM_GRID,
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

    print(f"\n  {'depth':>6}  {'min_split':>10}  {'Mean AUC':>10}  {'Std AUC':>10}")
    print(f"  {'-'*45}")
    for i, (_, row) in enumerate(cv_results_df.head(10).iterrows()):
        d      = row["param_classifier__max_depth"]
        ms     = row["param_classifier__min_samples_split"]
        mean   = row["mean_test_score"]
        std    = row["std_test_score"]
        marker = " <-- BEST" if i == 0 else ""
        print(f"  {d:>6}  {ms:>10}  {mean:>10.4f}  {std:>10.4f}{marker}")

    bp = grid_search.best_params_
    print(f"\n  Best max_depth        : {bp['classifier__max_depth']}")
    print(f"  Best min_samples_split: {bp['classifier__min_samples_split']}")
    print(f"  Best CV AUC           : {grid_search.best_score_:.4f}")

    # ------------------------------------------------------------------
    # 5. Full CV scoring with best config
    # ------------------------------------------------------------------
    print(f"\n[Step 4] Full CV scoring with best parameters")
    best_pipeline = build_decision_tree_pipeline(
        max_depth=bp["classifier__max_depth"],
        min_samples_split=bp["classifier__min_samples_split"],
        criterion=bp["classifier__criterion"],
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
    }, model_name="Decision Tree CART (10-fold CV)")

    # ------------------------------------------------------------------
    # 6. Final model (refitted by GridSearchCV on full training set)
    # ------------------------------------------------------------------
    print(f"\n[Step 5] Final model (refitted on full training set)")
    final_model = grid_search.best_estimator_

    # ------------------------------------------------------------------
    # 7. Tree structure info
    # ------------------------------------------------------------------
    tree_clf = final_model.named_steps["classifier"]
    print(f"\n  Tree structure:")
    print(f"    max_depth selected    : {bp['classifier__max_depth']}")
    print(f"    actual tree depth     : {tree_clf.get_depth()}")
    print(f"    number of leaves      : {tree_clf.get_n_leaves()}")
    print(f"    number of features    : {tree_clf.n_features_in_}")

    # Feature importances
    from config import FEATURES
    importances = tree_clf.feature_importances_
    feat_imp = sorted(zip(FEATURES, importances), key=lambda x: x[1], reverse=True)
    print(f"\n  Feature importances (Gini-based):")
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
        model_name="Decision Tree",
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        verbose=True,
    )

    # ------------------------------------------------------------------
    # 9. Save model
    # ------------------------------------------------------------------
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "decision_tree.joblib")
    joblib.dump(final_model, model_path)
    print(f"\n[Step 7] Model saved to: {model_path}")

    # ------------------------------------------------------------------
    # 10. Save results
    # ------------------------------------------------------------------
    os.makedirs(RESULT_DIR, exist_ok=True)
    results_path = os.path.join(RESULT_DIR, "decision_tree_metrics.csv")
    save_results_table([results], results_path)

    # ------------------------------------------------------------------
    # 11. Comparison with Logistic Regression
    # ------------------------------------------------------------------
    lr_results_path = os.path.join(RESULT_DIR, "logistic_regression_metrics.csv")
    if os.path.exists(lr_results_path):
        lr_df = pd.read_csv(lr_results_path)
        lr_auc = lr_df["ROC_AUC"].iloc[0]
        print(f"\n  COMPARISON (test set ROC-AUC so far):")
        print(f"    Logistic Regression : {lr_auc:.4f}")
        print(f"    Decision Tree       : {results['ROC_AUC']:.4f}")
        diff = results["ROC_AUC"] - lr_auc
        direction = "higher" if diff > 0 else "lower"
        print(f"    Difference          : {diff:+.4f} ({direction} than LR)")
        print(f"    NOTE: Do not draw conclusions yet -- XGBoost and MLP pending.")

    # ------------------------------------------------------------------
    # 12. Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"PHASE 7 COMPLETE -- DECISION TREE (CART)")
    print(f"{'='*65}")
    print(f"  Best max_depth        : {bp['classifier__max_depth']}")
    print(f"  Best min_samples_split: {bp['classifier__min_samples_split']}")
    print(f"  Actual tree depth     : {tree_clf.get_depth()}")
    print(f"  Number of leaves      : {tree_clf.get_n_leaves()}")
    print(f"  CV ROC-AUC (mean)     : {cv_scores['test_roc_auc'].mean():.4f}")
    print(f"  Test ROC-AUC          : {results['ROC_AUC']}")
    print(f"  Test Precision (c1)   : {results['Precision_Class1']}")
    print(f"  Test Recall (c1)      : {results['Recall_Class1']}")
    print(f"  Test F1 (c1)          : {results['F1_Class1']}")
    print(f"  Test Accuracy         : {results['Accuracy']}")
    print(f"  Model saved           : {model_path}")
    print(f"{'='*65}")

    return final_model, results


if __name__ == "__main__":
    train_decision_tree()
