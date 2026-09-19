"""
=============================================================================
PHASE 11 -- BAYESIAN HYPERPARAMETER OPTIMIZATION (OPTUNA)
=============================================================================
Project : GMSC Credit Scoring Research
File    : src/optuna_tuning.py
Purpose : Use Optuna TPE (Bayesian) search to optimise all 4 AI-driven
          models: XGBoost, LightGBM, Random Forest, MLP Neural Network.

Models tuned:
  1. XGBoost        -- gradient boosted trees
  2. LightGBM       -- histogram gradient boosting
  3. Random Forest  -- bagging ensemble
  4. MLP            -- feedforward neural network

Why Optuna beats GridSearchCV:
  - Learns from past trials (Bayesian) vs exhaustive brute-force
  - Searches continuous ranges, not just pre-defined values
  - 50 smart trials ~ equivalent to 300+ random grid points

Outputs (per model):
  - models/<model>_optuna.joblib
  - results/optuna_<model>_best_params.csv
  - plots/optuna_<model>_history.png
  - results/optuna_all_results.csv   (combined comparison)
=============================================================================
RUN COMMAND:
  cd credit-scoring-project
  python src/optuna_tuning.py
=============================================================================
"""

import sys, os
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")   # non-interactive backend -- required for background threads

import numpy as np
import pandas as pd
import joblib
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import matplotlib.pyplot as plt
from sklearn.model_selection  import StratifiedKFold, cross_val_score
from sklearn.pipeline         import Pipeline
from sklearn.ensemble         import RandomForestClassifier
from sklearn.neural_network   import MLPClassifier
from xgboost                  import XGBClassifier
from lightgbm                 import LGBMClassifier

from config             import RANDOM_STATE, MODEL_DIR, RESULT_DIR, PLOT_DIR
from data_preprocessing import load_data, clean_data, split_data, build_preprocessing_pipeline
from evaluation         import evaluate_model, save_results_table

# ---------------------------------------------------------------------------
N_TRIALS = 50   # Optuna trials per model
# ---------------------------------------------------------------------------

SKF5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


# =============================================================================
# OBJECTIVE FUNCTIONS
# =============================================================================

def xgboost_objective(trial, X, y, spw):
    params = dict(
        learning_rate    = trial.suggest_float("learning_rate",    0.005, 0.30,  log=True),
        max_depth        = trial.suggest_int(  "max_depth",        2,     7),
        n_estimators     = trial.suggest_int(  "n_estimators",     100,   500,   step=50),
        subsample        = trial.suggest_float("subsample",         0.60,  1.00),
        colsample_bytree = trial.suggest_float("colsample_bytree",  0.50,  1.00),
        min_child_weight = trial.suggest_int(  "min_child_weight",  1,     10),
        gamma            = trial.suggest_float("gamma",             0.00,  1.00),
        reg_alpha        = trial.suggest_float("reg_alpha",         0.00,  5.00),
        reg_lambda       = trial.suggest_float("reg_lambda",        1.00,  10.00),
        scale_pos_weight = spw,
        eval_metric="auc", use_label_encoder=False,
        random_state=RANDOM_STATE, verbosity=0, n_jobs=-1,
    )
    pipe = Pipeline([("pre", build_preprocessing_pipeline()),
                     ("clf", XGBClassifier(**params))])
    return cross_val_score(pipe, X, y, cv=SKF5, scoring="roc_auc", n_jobs=-1).mean()


def lightgbm_objective(trial, X, y, spw):
    params = dict(
        learning_rate     = trial.suggest_float("learning_rate",    0.005, 0.30,  log=True),
        num_leaves        = trial.suggest_int(  "num_leaves",       10,    150),
        n_estimators      = trial.suggest_int(  "n_estimators",     100,   500,   step=50),
        subsample         = trial.suggest_float("subsample",         0.60,  1.00),
        colsample_bytree  = trial.suggest_float("colsample_bytree",  0.50,  1.00),
        min_child_samples = trial.suggest_int(  "min_child_samples", 5,     50),
        reg_alpha         = trial.suggest_float("reg_alpha",         0.00,  5.00),
        reg_lambda        = trial.suggest_float("reg_lambda",        1.00,  10.00),
        scale_pos_weight  = spw,
        random_state=RANDOM_STATE, verbose=-1, n_jobs=-1,
    )
    pipe = Pipeline([("pre", build_preprocessing_pipeline()),
                     ("clf", LGBMClassifier(**params))])
    return cross_val_score(pipe, X, y, cv=SKF5, scoring="roc_auc", n_jobs=-1).mean()


def random_forest_objective(trial, X, y):
    params = dict(
        n_estimators      = trial.suggest_int(  "n_estimators",      50,   400,  step=50),
        max_depth         = trial.suggest_int(  "max_depth",          3,    20),
        min_samples_split = trial.suggest_int(  "min_samples_split",  5,    100),
        min_samples_leaf  = trial.suggest_int(  "min_samples_leaf",   2,    50),
        max_features      = trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        class_weight      = "balanced",
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    pipe = Pipeline([("pre", build_preprocessing_pipeline()),
                     ("clf", RandomForestClassifier(**params))])
    return cross_val_score(pipe, X, y, cv=SKF5, scoring="roc_auc", n_jobs=-1).mean()


def mlp_objective(trial, X, y):
    n_layers = trial.suggest_int("n_layers", 1, 3)
    layer_sizes = tuple(
        trial.suggest_int(f"n_units_l{i}", 16, 256, log=True)
        for i in range(n_layers)
    )
    params = dict(
        hidden_layer_sizes = layer_sizes,
        activation         = trial.suggest_categorical("activation", ["relu", "tanh"]),
        alpha              = trial.suggest_float("alpha",    1e-5, 1e-1, log=True),
        learning_rate_init = trial.suggest_float("lr_init",  1e-4, 1e-1, log=True),
        max_iter           = 100,
        early_stopping     = True,
        validation_fraction= 0.10,
        n_iter_no_change   = 10,
        random_state       = RANDOM_STATE,
    )
    pipe = Pipeline([("pre", build_preprocessing_pipeline()),
                     ("clf", MLPClassifier(**params))])
    return cross_val_score(pipe, X, y, cv=SKF5, scoring="roc_auc", n_jobs=-1).mean()


# =============================================================================
# PLOT HELPER
# =============================================================================

def plot_history(study, model_name, out_path):
    df = study.trials_dataframe()
    df = df[df["value"].notna()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    plt.style.use("seaborn-v0_8-whitegrid")

    ax1 = axes[0]
    ax1.scatter(df["number"], df["value"], alpha=0.45, color="#4a90e2", s=28, label="Trial AUC")
    ax1.plot(df["number"], df["value"].cummax(), color="#e94e77", linewidth=2.2, label="Best AUC")
    ax1.axhline(df["value"].max(), color="#2ca02c", linestyle="--", linewidth=1.2,
                label=f"Peak = {df['value'].max():.4f}")
    ax1.set_xlabel("Trial", fontsize=12)
    ax1.set_ylabel("5-Fold CV ROC-AUC", fontsize=12)
    ax1.set_title(f"{model_name} — Optuna Optimization History ({len(df)} trials)",
                  fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)

    try:
        imp = optuna.importance.get_param_importances(study)
        names = list(imp.keys())[:8]
        vals  = [imp[k] for k in names]
        ax2 = axes[1]
        colors = ["#e94e77" if v == max(vals) else "#4a90e2" for v in vals]
        ax2.barh(names[::-1], vals[::-1], color=colors[::-1])
        ax2.set_xlabel("Relative Importance", fontsize=12)
        ax2.set_title(f"{model_name} — Hyperparameter Importance", fontsize=13, fontweight="bold")
    except Exception:
        axes[1].text(0.5, 0.5, "Importance N/A", ha="center", va="center", fontsize=12)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Plot saved : {out_path}")


# =============================================================================
# TUNE ONE MODEL
# =============================================================================

def tune_model(name, objective_fn, model_builder_fn, X_train, X_test, y_train, y_test):
    """Run Optuna study, retrain best, evaluate, save model + params."""
    print(f"\n{'='*65}")
    print(f"  Tuning: {name}  ({N_TRIALS} Optuna trials)")
    print(f"{'='*65}")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        study_name=name.lower().replace(" ", "_"),
    )
    study.optimize(objective_fn, n_trials=N_TRIALS, show_progress_bar=True, n_jobs=1)

    best_params = study.best_params
    best_auc_cv = study.best_value

    print(f"\n  Best 5-fold CV AUC : {best_auc_cv:.4f}")
    print(f"  Best parameters:")
    for k, v in best_params.items():
        print(f"    {k:<25}: {v}")

    # Build & fit best pipeline on FULL training set
    print(f"\n  Retraining on full training set...")
    best_pipeline = model_builder_fn(best_params)
    best_pipeline.fit(X_train, y_train)

    y_pred = best_pipeline.predict(X_test)
    y_prob = best_pipeline.predict_proba(X_test)[:, 1]

    results = evaluate_model(
        model_name=f"{name} (Optuna)",
        y_true=y_test, y_pred=y_pred, y_prob=y_prob, verbose=True,
    )

    # Save model
    key   = name.lower().replace(" ", "_")
    mpath = os.path.join(MODEL_DIR, f"{key}_optuna.joblib")
    joblib.dump(best_pipeline, mpath)
    print(f"  Model saved : {mpath}")

    # Save best params CSV
    pdf = pd.DataFrame([best_params])
    pdf["cv_auc_5fold"] = best_auc_cv
    pdf.to_csv(os.path.join(RESULT_DIR, f"optuna_{key}_best_params.csv"), index=False)

    # Plot history
    plot_history(study, name, os.path.join(PLOT_DIR, f"optuna_{key}_history.png"))

    return results


# =============================================================================
# MAIN
# =============================================================================

def run_optuna_tuning():
    print("=" * 65)
    print("PHASE 11 -- BAYESIAN HYPERPARAMETER OPTIMIZATION")
    print("Models: XGBoost | LightGBM | Random Forest | MLP")
    print("=" * 65)
    print(f"  Algorithm : Optuna TPE (Tree-Structured Parzen Estimator)")
    print(f"  Trials    : {N_TRIALS} per model  ({N_TRIALS * 4} total)")
    print(f"  Inner CV  : 5-fold (speed) | Final eval: 30% holdout")

    # Load data once
    print("\n[Step 1] Loading data...")
    df_raw = load_data()
    df_clean, _ = clean_data(df_raw, verbose=False)
    X_train, X_test, y_train, y_test = split_data(df_clean)

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    spw = neg / pos
    print(f"  scale_pos_weight : {spw:.2f}")

    os.makedirs(MODEL_DIR,  exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR,   exist_ok=True)

    all_results = []

    # ------------------------------------------------------------------ XGBoost
    def build_xgb(p):
        return Pipeline([("pre", build_preprocessing_pipeline()),
                         ("clf", XGBClassifier(
                             **p, scale_pos_weight=spw,
                             eval_metric="auc", use_label_encoder=False,
                             random_state=RANDOM_STATE, verbosity=0, n_jobs=-1))])

    r = tune_model("XGBoost", lambda t: xgboost_objective(t, X_train, y_train, spw),
                   build_xgb, X_train, X_test, y_train, y_test)
    all_results.append(r)

    # ------------------------------------------------------------------ LightGBM
    def build_lgbm(p):
        return Pipeline([("pre", build_preprocessing_pipeline()),
                         ("clf", LGBMClassifier(
                             **p, scale_pos_weight=spw,
                             random_state=RANDOM_STATE, verbose=-1, n_jobs=-1))])

    r = tune_model("LightGBM", lambda t: lightgbm_objective(t, X_train, y_train, spw),
                   build_lgbm, X_train, X_test, y_train, y_test)
    all_results.append(r)

    # ------------------------------------------------------------------ Random Forest
    def build_rf(p):
        return Pipeline([("pre", build_preprocessing_pipeline()),
                         ("clf", RandomForestClassifier(
                             **p, class_weight="balanced",
                             random_state=RANDOM_STATE, n_jobs=-1))])

    r = tune_model("Random Forest", lambda t: random_forest_objective(t, X_train, y_train),
                   build_rf, X_train, X_test, y_train, y_test)
    all_results.append(r)

    # ------------------------------------------------------------------ MLP
    def build_mlp(p):
        p = dict(p)  # copy so we can mutate safely
        n_layers = p.pop("n_layers")
        # Reconstruct hidden_layer_sizes tuple from per-layer params
        hidden = tuple(p.pop(f"n_units_l{i}") for i in range(n_layers))
        activation = p.pop("activation")
        alpha      = p.pop("alpha")
        lr_init    = p.pop("lr_init")
        return Pipeline([("pre", build_preprocessing_pipeline()),
                         ("clf", MLPClassifier(
                             hidden_layer_sizes=hidden,
                             activation=activation,
                             alpha=alpha,
                             learning_rate_init=lr_init,
                             max_iter=100,
                             early_stopping=True,
                             validation_fraction=0.10,
                             n_iter_no_change=10,
                             random_state=RANDOM_STATE))])

    r = tune_model("MLP", lambda t: mlp_objective(t, X_train, y_train),
                   build_mlp, X_train, X_test, y_train, y_test)
    all_results.append(r)

    # ------------------------------------------------------------------ Summary
    print(f"\n{'='*65}")
    print(f"OPTUNA TUNING COMPLETE -- ALL 4 AI MODELS")
    print(f"{'='*65}")
    print(f"\n  {'Model':<28} {'Test ROC-AUC':>12} {'Recall':>8} {'Precision':>10} {'F1':>8}")
    print(f"  {'-'*68}")

    # Baseline (original GridSearch results)
    baselines = [
        ("XGBoost (GridSearch)",       "xgboost_metrics.csv"),
        ("LightGBM (GridSearch)",      "lightgbm_metrics.csv"),
        ("Random Forest (GridSearch)", "random_forest_metrics.csv"),
        ("MLP (GridSearch)",           "mlp_metrics.csv"),
    ]
    for label, fname in baselines:
        fp = os.path.join(RESULT_DIR, fname)
        if os.path.exists(fp):
            row = pd.read_csv(fp).iloc[0]
            print(f"  {label:<28} {row['ROC_AUC']:>12.4f} {row['Recall_Class1']:>8.4f} "
                  f"{row['Precision_Class1']:>10.4f} {row['F1_Class1']:>8.4f}")

    print(f"  {'─'*68}")
    for r in all_results:
        print(f"  {r['Model']:<28} {r['ROC_AUC']:>12.4f} {r['Recall_Class1']:>8.4f} "
              f"{r['Precision_Class1']:>10.4f} {r['F1_Class1']:>8.4f}")

    save_results_table(all_results, os.path.join(RESULT_DIR, "optuna_all_results.csv"))

    print(f"\n  Results CSV : results/optuna_all_results.csv")
    print(f"  Models saved: models/*_optuna.joblib")
    print(f"  Plots saved : plots/optuna_*_history.png")
    print(f"{'='*65}")

    return all_results


if __name__ == "__main__":
    run_optuna_tuning()
