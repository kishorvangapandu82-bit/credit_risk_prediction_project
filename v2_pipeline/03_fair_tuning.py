"""
=============================================================================
V2 PIPELINE -- STEP 3: FAIR TUNING ACROSS ALL 7 MODELS + WOE SCORECARD
=============================================================================
1. Documents exact search spaces in outputs_v2/tables/search_spaces.csv.
2. Tunes ALL 7 models using identical protocol:
   - Optuna TPE, 50 trials per model.
   - 5-fold Stratified CV ROC-AUC on training data.
   - Evaluates best model on untouched holdout test set.
3. Grid Search on XGBoost, LightGBM, and Random Forest (~48-50 evaluations each).
4. Traditional WOE Binning + Logistic Regression Scorecard baseline.
5. Saves:
   - outputs_v2/tables/search_spaces.csv
   - outputs_v2/tables/best_params.csv
   - outputs_v2/tables/tuning_comparison.csv
   - outputs_v2/tables/scorecard_metrics.csv
=============================================================================
"""

import os, sys, joblib, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Add src to path
sys.path.insert(0, os.path.abspath("src"))
from config import RANDOM_STATE, FEATURES
from data_preprocessing import load_data, clean_data, split_data, build_preprocessing_pipeline

TABLE_DIR = "outputs_v2/tables"
os.makedirs(TABLE_DIR, exist_ok=True)

# 1. Load data
print("[Tuning] Loading data...")
df_raw = load_data()
df_clean, _ = clean_data(df_raw, verbose=False)
X_train, X_test, y_train, y_test = split_data(df_clean)

SKF5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())  # ~13.96

# ---------------------------------------------------------------------------
# Document Search Spaces
# ---------------------------------------------------------------------------
search_spaces = [
    {
        "Model": "XGBoost",
        "Tuning_Protocol": "Optuna TPE (50 trials, 5-fold CV AUC)",
        "Hyperparameter": "learning_rate",
        "Type": "float (log)",
        "Range_or_Values": "[0.01, 0.20]"
    },
    {"Model": "XGBoost", "Tuning_Protocol": "Optuna TPE", "Hyperparameter": "max_depth", "Type": "int", "Range_or_Values": "[3, 8]"},
    {"Model": "XGBoost", "Tuning_Protocol": "Optuna TPE", "Hyperparameter": "n_estimators", "Type": "int", "Range_or_Values": "[100, 400]"},
    {"Model": "XGBoost", "Tuning_Protocol": "Optuna TPE", "Hyperparameter": "subsample", "Type": "float", "Range_or_Values": "[0.60, 1.00]"},
    {"Model": "XGBoost", "Tuning_Protocol": "Optuna TPE", "Hyperparameter": "colsample_bytree", "Type": "float", "Range_or_Values": "[0.50, 1.00]"},
    {"Model": "XGBoost", "Tuning_Protocol": "Optuna TPE", "Hyperparameter": "gamma", "Type": "float", "Range_or_Values": "[0.00, 1.00]"},
    {"Model": "XGBoost", "Tuning_Protocol": "Optuna TPE", "Hyperparameter": "reg_alpha", "Type": "float (log)", "Range_or_Values": "[1e-3, 10.0]"},
    {"Model": "XGBoost", "Tuning_Protocol": "Optuna TPE", "Hyperparameter": "reg_lambda", "Type": "float (log)", "Range_or_Values": "[1e-3, 10.0]"},
    {"Model": "XGBoost", "Tuning_Protocol": "Optuna TPE", "Hyperparameter": "scale_pos_weight", "Type": "constant", "Range_or_Values": f"{scale_pos_weight:.2f}"},

    {"Model": "LightGBM", "Tuning_Protocol": "Optuna TPE", "Hyperparameter": "learning_rate", "Type": "float (log)", "Range_or_Values": "[0.01, 0.20]"},
    {"Model": "LightGBM", "Tuning_Protocol": "Optuna TPE", "Hyperparameter": "num_leaves", "Type": "int", "Range_or_Values": "[15, 127]"},
    {"Model": "LightGBM", "Tuning_Protocol": "Optuna TPE", "Hyperparameter": "n_estimators", "Type": "int", "Range_or_Values": "[100, 400]"},
    {"Model": "LightGBM", "Tuning_Protocol": "Optuna TPE", "Hyperparameter": "min_child_samples", "Type": "int", "Range_or_Values": "[10, 100]"},
    {"Model": "LightGBM", "Tuning_Protocol": "Optuna TPE", "Hyperparameter": "reg_alpha", "Type": "float (log)", "Range_or_Values": "[1e-3, 10.0]"},
    {"Model": "LightGBM", "Tuning_Protocol": "Optuna TPE", "Hyperparameter": "reg_lambda", "Type": "float (log)", "Range_or_Values": "[1e-3, 10.0]"},
    {"Model": "LightGBM", "Tuning_Protocol": "Optuna TPE", "Hyperparameter": "scale_pos_weight", "Type": "constant", "Range_or_Values": f"{scale_pos_weight:.2f}"},

    {"Model": "Random Forest", "Tuning_Protocol": "Optuna TPE", "Hyperparameter": "n_estimators", "Type": "int", "Range_or_Values": "[100, 300]"},
    {"Model": "Random Forest", "Tuning_Protocol": "Optuna TPE", "Hyperparameter": "max_depth", "Type": "int", "Range_or_Values": "[6, 20]"},
    {"Model": "Random Forest", "Tuning_Protocol": "Optuna TPE", "Hyperparameter": "min_samples_split", "Type": "int", "Range_or_Values": "[10, 100]"},
    {"Model": "Random Forest", "Tuning_Protocol": "Optuna TPE", "Hyperparameter": "min_samples_leaf", "Type": "int", "Range_or_Values": "[5, 50]"},
    {"Model": "Random Forest", "Tuning_Protocol": "Optuna TPE", "Hyperparameter": "max_features", "Type": "categorical", "Range_or_Values": "{sqrt, log2}"},
    {"Model": "Random Forest", "Tuning_Protocol": "Optuna TPE", "Hyperparameter": "class_weight", "Type": "constant", "Range_or_Values": "balanced"},

    {"Model": "Decision Tree", "Tuning_Protocol": "Optuna TPE", "Hyperparameter": "max_depth", "Type": "int", "Range_or_Values": "[3, 12]"},
    {"Model": "Decision Tree", "Tuning_Protocol": "Optuna TPE", "Hyperparameter": "min_samples_split", "Type": "int", "Range_or_Values": "[10, 100]"},
    {"Model": "Decision Tree", "Tuning_Protocol": "Optuna TPE", "Hyperparameter": "min_samples_leaf", "Type": "int", "Range_or_Values": "[5, 50]"},
    {"Model": "Decision Tree", "Tuning_Protocol": "Optuna TPE", "Hyperparameter": "criterion", "Type": "categorical", "Range_or_Values": "{gini, entropy}"},
    {"Model": "Decision Tree", "Tuning_Protocol": "Optuna TPE", "Hyperparameter": "class_weight", "Type": "constant", "Range_or_Values": "balanced"},

    {"Model": "Logistic Regression", "Tuning_Protocol": "Optuna TPE", "Hyperparameter": "C", "Type": "float (log)", "Range_or_Values": "[1e-4, 10.0]"},
    {"Model": "Logistic Regression", "Tuning_Protocol": "Optuna TPE", "Hyperparameter": "class_weight", "Type": "constant", "Range_or_Values": "balanced"},

    {"Model": "Linear SVM", "Tuning_Protocol": "Optuna TPE", "Hyperparameter": "C", "Type": "float (log)", "Range_or_Values": "[1e-4, 10.0]"},
    {"Model": "Linear SVM", "Tuning_Protocol": "Optuna TPE", "Hyperparameter": "class_weight", "Type": "constant", "Range_or_Values": "balanced"},

    {"Model": "MLP", "Tuning_Protocol": "Optuna TPE", "Hyperparameter": "alpha", "Type": "float (log)", "Range_or_Values": "[1e-4, 1e-1]"},
    {"Model": "MLP", "Tuning_Protocol": "Optuna TPE", "Hyperparameter": "learning_rate_init", "Type": "float (log)", "Range_or_Values": "[1e-4, 1e-2]"},
    {"Model": "MLP", "Tuning_Protocol": "Optuna TPE", "Hyperparameter": "hidden_layer_sizes", "Type": "categorical", "Range_or_Values": "{(64, 32), (128, 64), (64,), (100,)}"},
]

pd.DataFrame(search_spaces).to_csv(os.path.join(TABLE_DIR, "search_spaces.csv"), index=False)
print("Saved outputs_v2/tables/search_spaces.csv")

# ---------------------------------------------------------------------------
# Optuna Tuning across ALL 7 Models
# ---------------------------------------------------------------------------
best_params_records = []
tuning_results = []

# 1. XGBoost
print("\n[Tuning 1/7] Optuna tuning XGBoost (50 trials)...")
def obj_xgb(trial):
    params = dict(
        learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
        max_depth        = trial.suggest_int("max_depth", 3, 8),
        n_estimators     = trial.suggest_int("n_estimators", 100, 400, step=50),
        subsample        = trial.suggest_float("subsample", 0.60, 1.00),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.50, 1.00),
        gamma            = trial.suggest_float("gamma", 0.00, 1.00),
        reg_alpha        = trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        reg_lambda       = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        scale_pos_weight = scale_pos_weight,
        eval_metric="auc", use_label_encoder=False,
        random_state=RANDOM_STATE, verbosity=0, n_jobs=-1,
    )
    pipe = Pipeline([("pre", build_preprocessing_pipeline()), ("clf", XGBClassifier(**params))])
    return cross_val_score(pipe, X_train, y_train, cv=SKF5, scoring="roc_auc", n_jobs=-1).mean()

study_xgb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study_xgb.optimize(obj_xgb, n_trials=50, timeout=600)
bp_xgb = study_xgb.best_params
bp_xgb["scale_pos_weight"] = scale_pos_weight
pipe_xgb_opt = Pipeline([("pre", build_preprocessing_pipeline()), ("clf", XGBClassifier(**bp_xgb, eval_metric="auc", use_label_encoder=False, random_state=RANDOM_STATE, verbosity=0, n_jobs=-1))])
pipe_xgb_opt.fit(X_train, y_train)
p_xgb_opt = pipe_xgb_opt.predict_proba(X_test)[:, 1]
auc_xgb_opt = roc_auc_score(y_test, p_xgb_opt)
print(f"  XGBoost Best CV: {study_xgb.best_value:.4f}, Test AUC: {auc_xgb_opt:.4f}")
best_params_records.append({"Model": "XGBoost", "Best_CV_AUC": round(study_xgb.best_value, 4), "Test_AUC": round(auc_xgb_opt, 4), "Best_Hyperparameters": str(bp_xgb)})

# 2. LightGBM
print("\n[Tuning 2/7] Optuna tuning LightGBM (50 trials)...")
def obj_lgb(trial):
    params = dict(
        learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
        num_leaves        = trial.suggest_int("num_leaves", 15, 127),
        n_estimators      = trial.suggest_int("n_estimators", 100, 400, step=50),
        min_child_samples = trial.suggest_int("min_child_samples", 10, 100),
        reg_alpha         = trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        reg_lambda        = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        scale_pos_weight  = scale_pos_weight,
        random_state=RANDOM_STATE, verbose=-1, n_jobs=-1,
    )
    pipe = Pipeline([("pre", build_preprocessing_pipeline()), ("clf", LGBMClassifier(**params))])
    return cross_val_score(pipe, X_train, y_train, cv=SKF5, scoring="roc_auc", n_jobs=-1).mean()

study_lgb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study_lgb.optimize(obj_lgb, n_trials=50, timeout=600)
bp_lgb = study_lgb.best_params
bp_lgb["scale_pos_weight"] = scale_pos_weight
pipe_lgb_opt = Pipeline([("pre", build_preprocessing_pipeline()), ("clf", LGBMClassifier(**bp_lgb, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1))])
pipe_lgb_opt.fit(X_train, y_train)
p_lgb_opt = pipe_lgb_opt.predict_proba(X_test)[:, 1]
auc_lgb_opt = roc_auc_score(y_test, p_lgb_opt)
print(f"  LightGBM Best CV: {study_lgb.best_value:.4f}, Test AUC: {auc_lgb_opt:.4f}")
best_params_records.append({"Model": "LightGBM", "Best_CV_AUC": round(study_lgb.best_value, 4), "Test_AUC": round(auc_lgb_opt, 4), "Best_Hyperparameters": str(bp_lgb)})

# 3. Random Forest
print("\n[Tuning 3/7] Optuna tuning Random Forest (50 trials)...")
def obj_rf(trial):
    params = dict(
        n_estimators      = trial.suggest_int("n_estimators", 100, 300, step=50),
        max_depth         = trial.suggest_int("max_depth", 6, 20),
        min_samples_split = trial.suggest_int("min_samples_split", 10, 100),
        min_samples_leaf  = trial.suggest_int("min_samples_leaf", 5, 50),
        max_features      = trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        class_weight      = "balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    pipe = Pipeline([("pre", build_preprocessing_pipeline()), ("clf", RandomForestClassifier(**params))])
    return cross_val_score(pipe, X_train, y_train, cv=SKF5, scoring="roc_auc", n_jobs=-1).mean()

study_rf = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study_rf.optimize(obj_rf, n_trials=50, timeout=600)
bp_rf = study_rf.best_params
bp_rf["class_weight"] = "balanced"
pipe_rf_opt = Pipeline([("pre", build_preprocessing_pipeline()), ("clf", RandomForestClassifier(**bp_rf, random_state=RANDOM_STATE, n_jobs=-1))])
pipe_rf_opt.fit(X_train, y_train)
p_rf_opt = pipe_rf_opt.predict_proba(X_test)[:, 1]
auc_rf_opt = roc_auc_score(y_test, p_rf_opt)
print(f"  Random Forest Best CV: {study_rf.best_value:.4f}, Test AUC: {auc_rf_opt:.4f}")
best_params_records.append({"Model": "Random Forest", "Best_CV_AUC": round(study_rf.best_value, 4), "Test_AUC": round(auc_rf_opt, 4), "Best_Hyperparameters": str(bp_rf)})

# 4. Decision Tree
print("\n[Tuning 4/7] Optuna tuning Decision Tree (50 trials)...")
def obj_dt(trial):
    params = dict(
        max_depth         = trial.suggest_int("max_depth", 3, 12),
        min_samples_split = trial.suggest_int("min_samples_split", 10, 100),
        min_samples_leaf  = trial.suggest_int("min_samples_leaf", 5, 50),
        criterion         = trial.suggest_categorical("criterion", ["gini", "entropy"]),
        class_weight      = "balanced",
        random_state=RANDOM_STATE,
    )
    pipe = Pipeline([("pre", build_preprocessing_pipeline()), ("clf", DecisionTreeClassifier(**params))])
    return cross_val_score(pipe, X_train, y_train, cv=SKF5, scoring="roc_auc", n_jobs=-1).mean()

study_dt = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study_dt.optimize(obj_dt, n_trials=50, timeout=300)
bp_dt = study_dt.best_params
bp_dt["class_weight"] = "balanced"
pipe_dt_opt = Pipeline([("pre", build_preprocessing_pipeline()), ("clf", DecisionTreeClassifier(**bp_dt, random_state=RANDOM_STATE))])
pipe_dt_opt.fit(X_train, y_train)
p_dt_opt = pipe_dt_opt.predict_proba(X_test)[:, 1]
auc_dt_opt = roc_auc_score(y_test, p_dt_opt)
print(f"  Decision Tree Best CV: {study_dt.best_value:.4f}, Test AUC: {auc_dt_opt:.4f}")
best_params_records.append({"Model": "Decision Tree", "Best_CV_AUC": round(study_dt.best_value, 4), "Test_AUC": round(auc_dt_opt, 4), "Best_Hyperparameters": str(bp_dt)})

# 5. Logistic Regression
print("\n[Tuning 5/7] Optuna tuning Logistic Regression (50 trials)...")
def obj_lr(trial):
    c_val = trial.suggest_float("C", 1e-4, 10.0, log=True)
    pipe = Pipeline([("pre", build_preprocessing_pipeline()),
                     ("clf", LogisticRegression(C=c_val, class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE))])
    return cross_val_score(pipe, X_train, y_train, cv=SKF5, scoring="roc_auc", n_jobs=-1).mean()

study_lr = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study_lr.optimize(obj_lr, n_trials=50, timeout=300)
bp_lr = study_lr.best_params
bp_lr["class_weight"] = "balanced"
pipe_lr_opt = Pipeline([("pre", build_preprocessing_pipeline()),
                        ("clf", LogisticRegression(C=bp_lr["C"], class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE))])
pipe_lr_opt.fit(X_train, y_train)
p_lr_opt = pipe_lr_opt.predict_proba(X_test)[:, 1]
auc_lr_opt = roc_auc_score(y_test, p_lr_opt)
print(f"  Logistic Regression Best CV: {study_lr.best_value:.4f}, Test AUC: {auc_lr_opt:.4f}")
best_params_records.append({"Model": "Logistic Regression", "Best_CV_AUC": round(study_lr.best_value, 4), "Test_AUC": round(auc_lr_opt, 4), "Best_Hyperparameters": str(bp_lr)})

# 6. Linear SVM
print("\n[Tuning 6/7] Optuna tuning Linear SVM (50 trials)...")
def obj_svm(trial):
    c_val = trial.suggest_float("C", 1e-4, 10.0, log=True)
    base_svc = LinearSVC(C=c_val, class_weight="balanced", dual="auto", max_iter=2000, random_state=RANDOM_STATE)
    cal_svc = CalibratedClassifierCV(estimator=base_svc, cv=3)
    pipe = Pipeline([("pre", build_preprocessing_pipeline()), ("clf", cal_svc)])
    return cross_val_score(pipe, X_train, y_train, cv=SKF5, scoring="roc_auc", n_jobs=-1).mean()

study_svm = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study_svm.optimize(obj_svm, n_trials=50, timeout=400)
bp_svm = study_svm.best_params
bp_svm["class_weight"] = "balanced"
base_svc = LinearSVC(C=bp_svm["C"], class_weight="balanced", dual="auto", max_iter=2000, random_state=RANDOM_STATE)
pipe_svm_opt = Pipeline([("pre", build_preprocessing_pipeline()), ("clf", CalibratedClassifierCV(estimator=base_svc, cv=3))])
pipe_svm_opt.fit(X_train, y_train)
p_svm_opt = pipe_svm_opt.predict_proba(X_test)[:, 1]
auc_svm_opt = roc_auc_score(y_test, p_svm_opt)
print(f"  Linear SVM Best CV: {study_svm.best_value:.4f}, Test AUC: {auc_svm_opt:.4f}")
best_params_records.append({"Model": "Linear SVM", "Best_CV_AUC": round(study_svm.best_value, 4), "Test_AUC": round(auc_svm_opt, 4), "Best_Hyperparameters": str(bp_svm)})

# 7. MLP Neural Network
print("\n[Tuning 7/7] Optuna tuning MLP (50 trials)...")
def obj_mlp(trial):
    alpha = trial.suggest_float("alpha", 1e-4, 1e-1, log=True)
    lr_init = trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True)
    h_idx = trial.suggest_categorical("h_idx", [0, 1, 2, 3])
    h_configs = [(64, 32), (128, 64), (64,), (100,)]
    hidden_size = h_configs[h_idx]
    
    pipe = Pipeline([("pre", build_preprocessing_pipeline()),
                     ("clf", MLPClassifier(hidden_layer_sizes=hidden_size, alpha=alpha, learning_rate_init=lr_init, max_iter=40, random_state=RANDOM_STATE, early_stopping=True, n_iter_no_change=5))])
    return cross_val_score(pipe, X_train, y_train, cv=SKF5, scoring="roc_auc", n_jobs=-1).mean()

study_mlp = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study_mlp.optimize(obj_mlp, n_trials=50, timeout=600)
bp_mlp = study_mlp.best_params
h_configs = [(64, 32), (128, 64), (64,), (100,)]
chosen_h = h_configs[bp_mlp["h_idx"]]
bp_mlp["hidden_layer_sizes"] = chosen_h

pipe_mlp_opt = Pipeline([("pre", build_preprocessing_pipeline()),
                         ("clf", MLPClassifier(hidden_layer_sizes=chosen_h, alpha=bp_mlp["alpha"], learning_rate_init=bp_mlp["learning_rate_init"], max_iter=50, random_state=RANDOM_STATE, early_stopping=True, n_iter_no_change=10))])
pipe_mlp_opt.fit(X_train, y_train)
p_mlp_opt = pipe_mlp_opt.predict_proba(X_test)[:, 1]
auc_mlp_opt = roc_auc_score(y_test, p_mlp_opt)
print(f"  MLP Best CV: {study_mlp.best_value:.4f}, Test AUC: {auc_mlp_opt:.4f}")
best_params_records.append({"Model": "MLP", "Best_CV_AUC": round(study_mlp.best_value, 4), "Test_AUC": round(auc_mlp_opt, 4), "Best_Hyperparameters": str(bp_mlp)})

bp_df = pd.DataFrame(best_params_records)
bp_df.to_csv(os.path.join(TABLE_DIR, "best_params.csv"), index=False)
print("\n=== Saved Best Parameters ===")
print(bp_df.to_string())

# ---------------------------------------------------------------------------
# Grid Search Comparison (XGBoost, LightGBM, Random Forest -- 48 evaluations)
# ---------------------------------------------------------------------------
print("\n[Grid Search] Running Grid Search on tree ensembles (48 evaluations each)...")
grid_results = []

# XGBoost Grid: 4 x 4 x 3 = 48 fits
xgb_grid = {
    "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "clf__max_depth": [3, 4, 5, 6],
    "clf__n_estimators": [100, 200, 300]
}
pipe_xgb_grid = Pipeline([("pre", build_preprocessing_pipeline()), ("clf", XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric="auc", use_label_encoder=False, random_state=RANDOM_STATE, verbosity=0, n_jobs=-1))])
gs_xgb = GridSearchCV(pipe_xgb_grid, xgb_grid, cv=SKF5, scoring="roc_auc", n_jobs=-1)
gs_xgb.fit(X_train, y_train)
p_xgb_gs = gs_xgb.best_estimator_.predict_proba(X_test)[:, 1]
auc_xgb_gs = roc_auc_score(y_test, p_xgb_gs)
grid_results.append({"Model": "XGBoost", "Grid_CV_AUC": round(gs_xgb.best_score_, 4), "Grid_Test_AUC": round(auc_xgb_gs, 4), "Optuna_CV_AUC": round(study_xgb.best_value, 4), "Optuna_Test_AUC": round(auc_xgb_opt, 4), "Optuna_Delta_Test_AUC": round(auc_xgb_opt - auc_xgb_gs, 4)})

# LightGBM Grid: 4 x 4 x 3 = 48 fits
lgb_grid = {
    "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "clf__num_leaves": [15, 31, 63, 127],
    "clf__n_estimators": [100, 200, 300]
}
pipe_lgb_grid = Pipeline([("pre", build_preprocessing_pipeline()), ("clf", LGBMClassifier(scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1))])
gs_lgb = GridSearchCV(pipe_lgb_grid, lgb_grid, cv=SKF5, scoring="roc_auc", n_jobs=-1)
gs_lgb.fit(X_train, y_train)
p_lgb_gs = gs_lgb.best_estimator_.predict_proba(X_test)[:, 1]
auc_lgb_gs = roc_auc_score(y_test, p_lgb_gs)
grid_results.append({"Model": "LightGBM", "Grid_CV_AUC": round(gs_lgb.best_score_, 4), "Grid_Test_AUC": round(auc_lgb_gs, 4), "Optuna_CV_AUC": round(study_lgb.best_value, 4), "Optuna_Test_AUC": round(auc_lgb_opt, 4), "Optuna_Delta_Test_AUC": round(auc_lgb_opt - auc_lgb_gs, 4)})

# Random Forest Grid: 3 x 4 x 4 = 48 fits
rf_grid = {
    "clf__n_estimators": [100, 200, 300],
    "clf__max_depth": [6, 10, 14, 18],
    "clf__min_samples_split": [10, 30, 50, 100]
}
pipe_rf_grid = Pipeline([("pre", build_preprocessing_pipeline()), ("clf", RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1))])
gs_rf = GridSearchCV(pipe_rf_grid, rf_grid, cv=SKF5, scoring="roc_auc", n_jobs=-1)
gs_rf.fit(X_train, y_train)
p_rf_gs = gs_rf.best_estimator_.predict_proba(X_test)[:, 1]
auc_rf_gs = roc_auc_score(y_test, p_rf_gs)
grid_results.append({"Model": "Random Forest", "Grid_CV_AUC": round(gs_rf.best_score_, 4), "Grid_Test_AUC": round(auc_rf_gs, 4), "Optuna_CV_AUC": round(study_rf.best_value, 4), "Optuna_Test_AUC": round(auc_rf_opt, 4), "Optuna_Delta_Test_AUC": round(auc_rf_opt - auc_rf_gs, 4)})

tuning_df = pd.DataFrame(grid_results)
tuning_df.to_csv(os.path.join(TABLE_DIR, "tuning_comparison.csv"), index=False)
print("\n=== Grid Search vs. Optuna TPE Comparison ===")
print(tuning_df.to_string())

# ---------------------------------------------------------------------------
# Traditional Scorecard Baseline (WOE Binning + Logistic Regression)
# ---------------------------------------------------------------------------
print("\n[Scorecard] Training traditional WOE Scorecard baseline...")

class WOETransformer:
    def __init__(self, bins=8):
        self.bins = bins
        self.woe_dict = {}
        self.bin_edges = {}

    def fit(self, X, y):
        self.woe_dict = {}
        self.bin_edges = {}
        total_good = (y == 0).sum()
        total_bad  = (y == 1).sum()

        for col in X.columns:
            s = X[col].copy()
            # Handle nulls
            is_na = s.isnull()
            valid_s = s[~is_na]
            
            # Discretize into quantiles
            _, edges = pd.qcut(valid_s, q=self.bins, retbins=True, duplicates="drop")
            self.bin_edges[col] = edges

            # Digitize
            bin_idx = np.digitize(valid_s, edges[1:-1])
            col_woe = {}
            for b in np.unique(bin_idx):
                mask = (s.index.isin(valid_s.index)) & (pd.Series(np.digitize(s, edges[1:-1]), index=s.index) == b)
                g = ((y == 0) & mask).sum()
                b_cnt = ((y == 1) & mask).sum()
                # Laplace smoothing
                p_g = (g + 0.5) / (total_good + 1.0)
                p_b = (b_cnt + 0.5) / (total_bad + 1.0)
                col_woe[b] = np.log(p_g / p_b)

            # Handle NA bin separately
            if is_na.sum() > 0:
                g_na = ((y == 0) & is_na).sum()
                b_na = ((y == 1) & is_na).sum()
                col_woe["NA"] = np.log(((g_na + 0.5) / (total_good + 1.0)) / ((b_na + 0.5) / (total_bad + 1.0)))
            else:
                col_woe["NA"] = 0.0

            self.woe_dict[col] = col_woe
        return self

    def transform(self, X):
        X_out = pd.DataFrame(index=X.index)
        for col in X.columns:
            edges = self.bin_edges[col]
            woe_map = self.woe_dict[col]
            s = X[col].copy()
            is_na = s.isnull()

            trans_col = np.zeros(len(s))
            valid_mask = ~is_na
            if valid_mask.sum() > 0:
                bin_idx = np.digitize(s[valid_mask], edges[1:-1])
                trans_col[valid_mask] = [woe_map.get(b, 0.0) for b in bin_idx]
            if is_na.sum() > 0:
                trans_col[is_na] = woe_map.get("NA", 0.0)
            X_out[col] = trans_col
        return X_out

woe = WOETransformer(bins=8).fit(X_train, y_train)
X_train_woe = woe.transform(X_train)
X_test_woe  = woe.transform(X_test)

scorecard = LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE)
scorecard.fit(X_train_woe, y_train)

p_scorecard = scorecard.predict_proba(X_test_woe)[:, 1]
pred_scorecard = scorecard.predict(X_test_woe)

sc_auc  = roc_auc_score(y_test, p_scorecard)
sc_f1   = f1_score(y_test, pred_scorecard)
sc_rec  = recall_score(y_test, pred_scorecard)
sc_prec = precision_score(y_test, pred_scorecard)
sc_acc  = accuracy_score(y_test, pred_scorecard)

scorecard_df = pd.DataFrame([{
    "Model": "WOE Logistic Scorecard",
    "ROC_AUC": round(sc_auc, 4),
    "Gini": round(2.0 * sc_auc - 1.0, 4),
    "Recall": round(sc_rec, 4),
    "Precision": round(sc_prec, 4),
    "F1": round(sc_f1, 4),
    "Accuracy": round(sc_acc, 4),
    "Binning_Strategy": "8 Quantile Bins with Missing Category",
    "Weighting": "balanced"
}])

scorecard_df.to_csv(os.path.join(TABLE_DIR, "scorecard_metrics.csv"), index=False)
print("\n=== Classical Scorecard (WOE + LogReg) Holdout Metrics ===")
print(scorecard_df.to_string())

print("\nStep 3 Complete!")
