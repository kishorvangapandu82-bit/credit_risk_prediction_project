"""
=============================================================================
PHASE 9 -- MULTI-LAYER PERCEPTRON (MLP) NEURAL NETWORK TRAINING
=============================================================================
Project : GMSC Credit Scoring Research
File    : src/train_mlp.py
Purpose : Train Multi-Layer Perceptron (MLP) with stratified 10-fold CV.
          Tune regularization alpha and initial learning rate.
          Evaluate best model on untouched 30% test set.

Why MLP?
  - Fully connected feedforward artificial neural network
  - Learns hierarchical non-linear representations across multiple dense layers
  - Tests whether deep representations outperform tree ensembles (XGBoost)
    and shallow decision boundaries (Logistic Regression / CART)

Architecture (Fixed per methodology):
  - Input Layer   : 10 standardized features
  - Hidden Layer 1: 64 neurons (ReLU activation)
  - Hidden Layer 2: 32 neurons (ReLU activation)
  - Output Layer  : Sigmoid / binary classification (1 output unit)
  - Max Epochs    : 50
  - Optimizer     : Adam
  - Early Stopping: True (validation_fraction=0.1, n_iter_no_change=10)
  - Imbalance     : Sample weighting (scale_pos_weight = count(0)/count(1) ~ 13.96)

Hyperparameters to Tune:
  - alpha              : L2 regularization penalty [0.0001, 0.001, 0.01]
  - learning_rate_init : Adam initial step size [0.001, 0.005]

Pipeline Structure (leakage-safe):
  [Imputer + Scaler fitted strictly per fold] --> [MLPClassifier]

=============================================================================
RUN COMMAND:
  cd credit-scoring-project
  python src/train_mlp.py
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
from sklearn.neural_network   import MLPClassifier

from config               import (
    RANDOM_STATE, CV_FOLDS, MODEL_DIR, RESULT_DIR,
    MLP_EPOCHS, MLP_HIDDEN_1, MLP_HIDDEN_2, MLP_ACTIVATION
)
from data_preprocessing   import load_data, clean_data, split_data
from models               import build_mlp_pipeline
from evaluation           import evaluate_model, print_cv_results, save_results_table


# ---------------------------------------------------------------------------
# Hyperparameter search space
# ---------------------------------------------------------------------------
MLP_PARAM_GRID = {
    "classifier__alpha":              [0.0001, 0.001, 0.01],
    "classifier__learning_rate_init": [0.001, 0.005],
}


def f1_class1(y_true, y_pred):
    return f1_score(y_true, y_pred, pos_label=1, zero_division=0)

def recall_class1(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=1, zero_division=0)

def precision_class1(y_true, y_pred):
    return precision_score(y_true, y_pred, pos_label=1, zero_division=0)


def train_mlp():

    print("=" * 65)
    print("PHASE 9 -- MULTI-LAYER PERCEPTRON (MLP) TRAINING")
    print("=" * 65)
    print(f"\n  Model               : Multi-Layer Perceptron (MLP)")
    print(f"  Architecture        : Input(10) -> Dense({MLP_HIDDEN_1}, ReLU) -> Dense({MLP_HIDDEN_2}, ReLU) -> Output(1)")
    print(f"  Max Epochs          : {MLP_EPOCHS}")
    print(f"  Early Stopping      : True (patience=10)")
    print(f"  alpha grid          : {MLP_PARAM_GRID['classifier__alpha']}")
    print(f"  lr_init grid        : {MLP_PARAM_GRID['classifier__learning_rate_init']}")
    print(f"  CV strategy         : Stratified {CV_FOLDS}-fold (training data only)")
    print(f"  Scoring             : ROC-AUC")
    
    n_combinations = (
        len(MLP_PARAM_GRID["classifier__alpha"])
        * len(MLP_PARAM_GRID["classifier__learning_rate_init"])
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

    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = neg_count / pos_count
    
    # Generate sample weights strictly on training labels
    sample_weights_train = np.where(y_train == 1, scale_pos_weight, 1.0)

    print(f"  Training set        : {X_train.shape} | Class 1: {pos_count} ({y_train.mean()*100:.2f}%)")
    print(f"  Test set            : {X_test.shape}  | Class 1: {int(y_test.sum())}  ({y_test.mean()*100:.2f}%)")
    print(f"  sample_weight (c1)  : {scale_pos_weight:.2f} (applied to training loss)")

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

    base_pipeline, _ = build_mlp_pipeline()

    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=MLP_PARAM_GRID,
        cv=skf,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    grid_search.fit(X_train, y_train, classifier__sample_weight=sample_weights_train)

    # ------------------------------------------------------------------
    # 4. CV results table
    # ------------------------------------------------------------------
    print(f"\n[Step 3] GridSearchCV results (sorted by mean AUC)")
    cv_results_df = pd.DataFrame(grid_search.cv_results_)
    cv_results_df = cv_results_df.sort_values("mean_test_score", ascending=False)

    print(f"\n  {'alpha':>10}  {'lr_init':>10}  {'Mean AUC':>10}  {'Std AUC':>10}")
    print(f"  {'-'*46}")
    for i, (_, row) in enumerate(cv_results_df.iterrows()):
        alpha  = row["param_classifier__alpha"]
        lr_in  = row["param_classifier__learning_rate_init"]
        mean   = row["mean_test_score"]
        std    = row["std_test_score"]
        marker = " <-- BEST" if i == 0 else ""
        print(f"  {alpha:>10}  {lr_in:>10}  {mean:>10.4f}  {std:>10.4f}{marker}")

    bp = grid_search.best_params_
    print(f"\n  Best alpha             : {bp['classifier__alpha']}")
    print(f"  Best learning_rate_init: {bp['classifier__learning_rate_init']}")
    print(f"  Best CV ROC-AUC        : {grid_search.best_score_:.4f}")

    # ------------------------------------------------------------------
    # 5. Full CV scoring with best config
    # ------------------------------------------------------------------
    print(f"\n[Step 4] Full CV scoring with best parameters")
    best_pipeline, _ = build_mlp_pipeline()
    best_pipeline.named_steps["classifier"].set_params(
        alpha=bp["classifier__alpha"],
        learning_rate_init=bp["classifier__learning_rate_init"]
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
        params={"classifier__sample_weight": sample_weights_train}
    )

    print_cv_results({
        "ROC-AUC"            : cv_scores["test_roc_auc"],
        "F1 (class 1)"       : cv_scores["test_f1_class1"],
        "Recall (class 1)"   : cv_scores["test_recall_c1"],
        "Precision (class 1)": cv_scores["test_precision_c1"],
        "Accuracy"           : cv_scores["test_accuracy"],
    }, model_name="MLP Neural Network (10-fold CV)")

    # ------------------------------------------------------------------
    # 6. Final model (refitted by GridSearchCV on full training set)
    # ------------------------------------------------------------------
    print(f"\n[Step 5] Final model inspection")
    final_model = grid_search.best_estimator_
    mlp_clf = final_model.named_steps["classifier"]
    print(f"  Trained iterations : {mlp_clf.n_iter_} / {MLP_EPOCHS}")
    print(f"  Final training loss: {mlp_clf.loss_:.4f}")

    # ------------------------------------------------------------------
    # 7. Evaluate on untouched test set
    # ------------------------------------------------------------------
    print(f"\n[Step 6] Evaluating on untouched 30% test set")
    y_pred = final_model.predict(X_test)
    y_prob = final_model.predict_proba(X_test)[:, 1]

    results = evaluate_model(
        model_name="MLP Neural Network",
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        verbose=True,
    )

    # ------------------------------------------------------------------
    # 8. Save model
    # ------------------------------------------------------------------
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "mlp.joblib")
    joblib.dump(final_model, model_path)
    print(f"\n[Step 7] Model saved to: {model_path}")

    # ------------------------------------------------------------------
    # 9. Save results
    # ------------------------------------------------------------------
    os.makedirs(RESULT_DIR, exist_ok=True)
    results_path = os.path.join(RESULT_DIR, "mlp_metrics.csv")
    save_results_table([results], results_path)

    # ------------------------------------------------------------------
    # 10. Complete 4-Model Comparative Table
    # ------------------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"FINAL 4-MODEL COMPARISON (Untouched 30% Test Set)")
    print(f"{'='*65}")
    print(f"  {'Model':<24} {'ROC-AUC':>10} {'Recall':>10} {'Precision':>10} {'F1':>10} {'Accuracy':>10}")
    print(f"  {'-'*76}")

    all_models = []
    for m_file in ["logistic_regression_metrics.csv", "decision_tree_metrics.csv", "xgboost_metrics.csv"]:
        fp = os.path.join(RESULT_DIR, m_file)
        if os.path.exists(fp):
            df = pd.read_csv(fp)
            all_models.append(df.iloc[0])
    all_models.append(pd.Series(results))

    for m in all_models:
        print(f"  {m['Model']:<24} {m['ROC_AUC']:>10.4f} {m['Recall_Class1']:>10.4f} {m['Precision_Class1']:>10.4f} {m['F1_Class1']:>10.4f} {m['Accuracy']:>10.4f}")

    # ------------------------------------------------------------------
    # 11. Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"PHASE 9 COMPLETE -- MULTI-LAYER PERCEPTRON (MLP)")
    print(f"{'='*65}")
    print(f"  Best alpha            : {bp['classifier__alpha']}")
    print(f"  Best lr_init          : {bp['classifier__learning_rate_init']}")
    print(f"  Trained iterations    : {mlp_clf.n_iter_}")
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
    train_mlp()
