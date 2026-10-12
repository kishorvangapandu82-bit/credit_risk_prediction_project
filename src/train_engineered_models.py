"""
=============================================================================
TRAIN & BENCHMARK MODELS ON ENGINEERED FEATURES (PHASE 2)
=============================================================================
Project : GMSC Credit Scoring Research
File    : src/train_engineered_models.py
Purpose : Train LightGBM, XGBoost, and Random Forest on 18 features (10 baseline + 8 engineered),
          evaluate on the 45,000-sample test set, compare performance against baseline,
          and generate ROC comparison plots.
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, recall_score, precision_score, f1_score,
    accuracy_score, roc_curve
)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# Ensure src/ is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RANDOM_STATE, RESULT_DIR, PLOT_DIR, MODEL_DIR
from feature_engineering import prepare_engineered_splits


def train_and_evaluate_engineered():
    print("=" * 70)
    print("PHASE 2: TRAINING MODELS ON 18 ENGINEERED FINANCIAL FEATURES")
    print("=" * 70)

    # 1. Load data with engineered features
    X_train, X_test, y_train, y_test = prepare_engineered_splits()
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

    # Models configuration with Bayesian-informed parameters
    models = {
        "LightGBM (Engineered)": LGBMClassifier(
            learning_rate=0.0453,
            num_leaves=14,
            n_estimators=200,
            subsample=0.9208,
            colsample_bytree=0.5593,
            min_child_samples=41,
            reg_alpha=2.0195,
            reg_lambda=7.8228,
            scale_pos_weight=scale_pos,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
        "XGBoost (Engineered)": XGBClassifier(
            learning_rate=0.0527,
            max_depth=4,
            n_estimators=250,
            subsample=0.7867,
            colsample_bytree=0.6418,
            min_child_weight=9,
            gamma=0.8114,
            reg_alpha=4.1148,
            reg_lambda=8.0420,
            scale_pos_weight=scale_pos,
            eval_metric="auc",
            use_label_encoder=False,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        ),
        "Random Forest (Engineered)": RandomForestClassifier(
            n_estimators=300,
            max_depth=11,
            min_samples_split=91,
            min_samples_leaf=29,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    # Baseline scores are now the BAYESIAN OPTIMIZATION (Optuna TPE) baselines (10 raw features)
    bayesian_baselines = {
        "LightGBM": {"ROC_AUC": 0.8675, "Recall": 0.7776, "Precision": 0.2157, "F1": 0.3377, "Accuracy": 0.7962},
        "XGBoost": {"ROC_AUC": 0.8673, "Recall": 0.7746, "Precision": 0.2169, "F1": 0.3389, "Accuracy": 0.7980},
        "Random Forest": {"ROC_AUC": 0.8655, "Recall": 0.7726, "Precision": 0.2182, "F1": 0.3403, "Accuracy": 0.7998},
    }

    results = []
    roc_curves = {}
    probs_dict = {}

    for name, model in models.items():
        base_name = name.split(" (")[0]
        print(f"\n---> Training {name} on {X_train.shape[1]} features...")
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        probs_dict[base_name] = y_prob
        y_pred = (y_prob >= 0.5).astype(int)

        auc = roc_auc_score(y_test, y_prob)
        rec = recall_score(y_test, y_pred, pos_label=1)
        prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        acc = accuracy_score(y_test, y_pred)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_curves[name] = (fpr, tpr, auc)

        base = bayesian_baselines[base_name]
        auc_delta = auc - base["ROC_AUC"]

        print(f"  Result for {name}:")
        print(f"    ROC-AUC   : {auc:.4f}  (Bayesian Baseline: {base['ROC_AUC']:.4f}, Delta: {auc_delta:+.4f})")
        print(f"    Recall    : {rec:.4f}  (Bayesian Baseline: {base['Recall']:.4f})")
        print(f"    Precision : {prec:.4f}  (Bayesian Baseline: {base['Precision']:.4f})")
        print(f"    F1-Score  : {f1:.4f}  (Bayesian Baseline: {base['F1']:.4f})")
        print(f"    Accuracy  : {acc:.4f}  (Bayesian Baseline: {base['Accuracy']:.4f})")

        results.append({
            "Model": name,
            "Baseline_Features": 10,
            "Engineered_Features": 18,
            "Baseline_Bayesian_ROC_AUC": base["ROC_AUC"],
            "Engineered_ROC_AUC": round(auc, 4),
            "AUC_Improvement": round(auc_delta, 4),
            "Recall_Class1": round(rec, 4),
            "Precision_Class1": round(prec, 4),
            "F1_Class1": round(f1, 4),
            "Accuracy": round(acc, 4),
        })

    df_res = pd.DataFrame(results)
    out_csv = os.path.join(RESULT_DIR, "feature_engineering_comparison.csv")
    df_res.to_csv(out_csv, index=False)
    print(f"\n[Done] Comparison saved to: {out_csv}")
    print(df_res.to_string())

    # Generate Visualization: ROC Comparison Plot
    plt.figure(figsize=(9, 7), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    colors = {
        "LightGBM (Engineered)": "#10B981",
        "XGBoost (Engineered)": "#3B82F6",
        "Random Forest (Engineered)": "#F59E0B",
    }

    for name, (fpr, tpr, auc) in roc_curves.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})", color=colors.get(name, "black"), lw=2.2)

    # Reference lines for baselines
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (AUC = 0.5000)")

    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold")
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=12, fontweight="bold")
    plt.title("ROC Curves: 18 Domain-Engineered Features", fontsize=14, fontweight="bold", pad=15)
    plt.legend(loc="lower right", fontsize=11, frameon=True)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()

    out_plot = os.path.join(PLOT_DIR, "feature_engineering_roc_comparison.png")
    plt.savefig(out_plot, dpi=300)
    plt.close()
    print(f"[Done] ROC plot saved to: {out_plot}")


if __name__ == "__main__":
    train_and_evaluate_engineered()
