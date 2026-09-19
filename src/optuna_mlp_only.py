"""
Standalone MLP-only Optuna tuning.
Runs after XGBoost/LightGBM/RF already completed in optuna_tuning.py.
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import joblib
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline        import Pipeline
from sklearn.neural_network  import MLPClassifier

from config             import RANDOM_STATE, MODEL_DIR, RESULT_DIR, PLOT_DIR
from data_preprocessing import load_data, clean_data, split_data, build_preprocessing_pipeline
from evaluation         import evaluate_model, save_results_table

N_TRIALS = 50
SKF5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


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


def plot_history(study, out_path):
    df = study.trials_dataframe()
    df = df[df["value"].notna()]
    plt.figure(figsize=(10, 4), dpi=150)
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.scatter(df["number"], df["value"], alpha=0.45, color="#9467bd", s=28, label="Trial AUC")
    plt.plot(df["number"], df["value"].cummax(), color="#e94e77", linewidth=2.2, label="Best AUC")
    plt.axhline(df["value"].max(), color="#2ca02c", linestyle="--", linewidth=1.2,
                label=f"Peak = {df['value'].max():.4f}")
    plt.xlabel("Trial"); plt.ylabel("5-Fold CV ROC-AUC")
    plt.title(f"MLP — Optuna Optimization History ({len(df)} trials)", fontweight="bold")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()
    print(f"  Plot saved: {out_path}")


def run():
    print("=" * 65)
    print("MLP OPTUNA TUNING (standalone fix)")
    print("=" * 65)

    df_raw = load_data()
    df_clean, _ = clean_data(df_raw, verbose=False)
    X_train, X_test, y_train, y_test = split_data(df_clean)

    os.makedirs(MODEL_DIR,  exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR,   exist_ok=True)

    print(f"\n  Tuning MLP with {N_TRIALS} Optuna trials...")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        study_name="mlp_credit_scoring",
    )
    study.optimize(
        lambda t: mlp_objective(t, X_train, y_train),
        n_trials=N_TRIALS, show_progress_bar=True, n_jobs=1,
    )

    bp = study.best_params
    best_cv = study.best_value
    print(f"\n  Best 5-fold CV AUC : {best_cv:.4f}")
    print(f"  Best parameters    : {bp}")

    # Reconstruct properly
    n_layers = bp["n_layers"]
    hidden   = tuple(bp[f"n_units_l{i}"] for i in range(n_layers))
    activation = bp["activation"]
    alpha      = bp["alpha"]
    lr_init    = bp["lr_init"]

    print(f"\n  Retraining best MLP on full training set...")
    best_pipe = Pipeline([("pre", build_preprocessing_pipeline()),
                          ("clf", MLPClassifier(
                              hidden_layer_sizes=hidden,
                              activation=activation,
                              alpha=alpha,
                              learning_rate_init=lr_init,
                              max_iter=100,
                              early_stopping=True,
                              validation_fraction=0.10,
                              n_iter_no_change=10,
                              random_state=RANDOM_STATE,
                          ))])
    best_pipe.fit(X_train, y_train)

    y_pred = best_pipe.predict(X_test)
    y_prob = best_pipe.predict_proba(X_test)[:, 1]
    results = evaluate_model("MLP (Optuna)", y_test, y_pred, y_prob, verbose=True)

    mpath = os.path.join(MODEL_DIR, "mlp_optuna.joblib")
    joblib.dump(best_pipe, mpath)
    print(f"  Model saved: {mpath}")

    pdf = pd.DataFrame([bp])
    pdf["cv_auc_5fold"] = best_cv
    pdf.to_csv(os.path.join(RESULT_DIR, "optuna_mlp_best_params.csv"), index=False)

    plot_history(study, os.path.join(PLOT_DIR, "optuna_mlp_history.png"))

    # ── Combined final comparison ──────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"OPTUNA RESULTS -- ALL 4 AI MODELS (GridSearch vs Optuna)")
    print(f"{'='*65}")
    print(f"  {'Model':<30} {'Base AUC':>10} {'Optuna AUC':>12} {'Gain':>8}")
    print(f"  {'-'*62}")

    pairs = [
        ("XGBoost",       "xgboost_metrics.csv",       "optuna_xgboost_best_params.csv"),
        ("LightGBM",      "lightgbm_metrics.csv",       "optuna_lightgbm_best_params.csv"),
        ("Random Forest", "random_forest_metrics.csv",  "optuna_random_forest_best_params.csv"),
        ("MLP",           "mlp_metrics.csv",            "optuna_mlp_best_params.csv"),
    ]
    for name, base_f, opt_f in pairs:
        bp_path = os.path.join(RESULT_DIR, base_f)
        op_path = os.path.join(RESULT_DIR, opt_f)
        if os.path.exists(bp_path) and os.path.exists(op_path):
            base_auc  = pd.read_csv(bp_path).iloc[0]["ROC_AUC"]
            optuna_auc = pd.read_csv(op_path).iloc[0]["cv_auc_5fold"]
            gain = optuna_auc - base_auc
            print(f"  {name:<30} {base_auc:>10.4f} {optuna_auc:>12.4f} {gain:>+8.4f}")

    # Also save MLP result to combined CSV alongside others
    prev = os.path.join(RESULT_DIR, "optuna_all_results.csv")
    if os.path.exists(prev):
        existing = pd.read_csv(prev)
        combined = pd.concat([existing, pd.DataFrame([results])], ignore_index=True)
    else:
        combined = pd.DataFrame([results])
    combined.to_csv(prev, index=False)
    print(f"\n  Final results saved: results/optuna_all_results.csv")
    print(f"{'='*65}")


if __name__ == "__main__":
    run()
