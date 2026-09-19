"""
=============================================================================
PHASE 10A -- COMPARATIVE VISUALIZATIONS
=============================================================================
Project : GMSC Credit Scoring Research
File    : src/visualize_results.py
Purpose : Generate publication-quality comparative evaluation figures:
          1. Combined ROC Curves with AUC in legend
          2. Precision-Recall Curves with Average Precision (AP)
          3. Calibration Curves (Reliability Diagrams)
          4. Grouped Metrics Comparison Bar Chart
          5. Combined Confusion Matrices (2x2 grid)

All figures are saved at 300 DPI in plots/.
=============================================================================
RUN COMMAND:
  cd credit-scoring-project
  python src/visualize_results.py
=============================================================================
"""

import sys, os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.calibration import calibration_curve

from config import MODEL_DIR, RESULT_DIR, PLOT_DIR
from data_preprocessing import load_data, clean_data, split_data


# Model configuration dictionary
MODELS_CONFIG = [
    {
        "name": "Logistic Regression",
        "file": "logistic_regression.joblib",
        "color": "#1f77b4",   # Muted blue
        "linestyle": "--",
    },
    {
        "name": "Decision Tree (CART)",
        "file": "decision_tree.joblib",
        "color": "#ff7f0e",   # Warm orange
        "linestyle": "-.",
    },
    {
        "name": "MLP Neural Network",
        "file": "mlp.joblib",
        "color": "#9467bd",   # Purple
        "linestyle": ":",
    },
    {
        "name": "XGBoost",
        "file": "xgboost.joblib",
        "color": "#2ca02c",   # Forest green
        "linestyle": "-",
    },
    {
        "name": "LightGBM",
        "file": "lightgbm.joblib",
        "color": "#17becf",   # Cyan / teal
        "linestyle": "-",
    },
    {
        "name": "Support Vector Machine",
        "file": "svm.joblib",
        "color": "#d62728",   # Crimson red
        "linestyle": "--",
    },
    {
        "name": "Random Forest",
        "file": "random_forest.joblib",
        "color": "#8c564b",   # Brown / chestnut
        "linestyle": "-.",
    },
]


def load_models_and_predict():
    """Load test dataset and generate predictions for all trained models."""
    print("\n[Step 1] Loading test data...")
    df_raw = load_data()
    df_clean, _ = clean_data(df_raw, verbose=False)
    _, X_test, _, y_test = split_data(df_clean)

    predictions = {}
    print("\n[Step 2] Generating predictions from saved models...")
    for m in MODELS_CONFIG:
        model_path = os.path.join(MODEL_DIR, m["file"])
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        print(f"  Loading {m['name']} from {m['file']}...")
        model = joblib.load(model_path)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        predictions[m["name"]] = {
            "y_prob": y_prob,
            "y_pred": y_pred,
            "color": m["color"],
            "linestyle": m["linestyle"],
        }

    return y_test, predictions


def plot_roc_curves(y_test, predictions):
    """Plot combined ROC Curves for all models."""
    print("\n[Step 3] Plotting ROC Curves...")
    plt.figure(figsize=(9, 7), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid")

    for name, data in predictions.items():
        fpr, tpr, _ = roc_curve(y_test, data["y_prob"])
        roc_auc = auc(fpr, tpr)
        plt.plot(
            fpr, tpr,
            color=data["color"],
            linestyle=data["linestyle"],
            linewidth=2.4,
            label=f"{name} (AUC = {roc_auc:.4f})"
        )

    plt.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1.5, label="Random Guess (AUC = 0.5000)")
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=13, labelpad=8)
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=13, labelpad=8)
    plt.title("Receiver Operating Characteristic (ROC) Curves\nAll Models — Overall Performance (N = 45,000)", fontsize=14, fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=11)
    plt.tight_layout()

    out_path = os.path.join(PLOT_DIR, "roc_curves_comparison.png")
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_precision_recall_curves(y_test, predictions):
    """Plot Precision-Recall Curves."""
    print("\n[Step 4] Plotting Precision-Recall Curves...")
    plt.figure(figsize=(9, 7), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid")

    baseline_rate = y_test.mean()

    for name, data in predictions.items():
        prec, rec, _ = precision_recall_curve(y_test, data["y_prob"])
        ap = average_precision_score(y_test, data["y_prob"])
        plt.plot(
            rec, prec,
            color=data["color"],
            linestyle=data["linestyle"],
            linewidth=2.4,
            label=f"{name} (AP = {ap:.4f})"
        )

    plt.axhline(y=baseline_rate, color="gray", linestyle=":", linewidth=1.5, label=f"No-Skill Baseline ({baseline_rate*100:.2f}%)")
    plt.xlim([0.0, 1.02])
    plt.ylim([0.0, 0.65])
    plt.xlabel("Recall (Default Detection Rate)", fontsize=13, labelpad=8)
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=13, labelpad=8)
    plt.title("Precision-Recall (PR) Curves under Imbalanced Class Distribution\nMinority Class Ratio = 6.68%", fontsize=14, fontweight="bold", pad=12)
    plt.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=11)
    plt.tight_layout()

    out_path = os.path.join(PLOT_DIR, "precision_recall_comparison.png")
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_calibration_curves(y_test, predictions):
    """Plot Calibration Curves (Reliability Diagrams)."""
    print("\n[Step 5] Plotting Calibration Curves...")
    plt.figure(figsize=(9, 7), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid")

    plt.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1.5, label="Perfect Calibration")

    for name, data in predictions.items():
        prob_true, prob_pred = calibration_curve(y_test, data["y_prob"], n_bins=10)
        plt.plot(
            prob_pred, prob_true,
            marker="o",
            markersize=5,
            color=data["color"],
            linestyle=data["linestyle"],
            linewidth=2.0,
            label=f"{name}"
        )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("Mean Predicted Probability", fontsize=13, labelpad=8)
    plt.ylabel("Fraction of True Defaults (Empirical Probability)", fontsize=13, labelpad=8)
    plt.title("Probability Calibration (Reliability Diagram)\n10 Bins — Overall Model Performance", fontsize=14, fontweight="bold", pad=12)
    plt.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=11)
    plt.tight_layout()

    out_path = os.path.join(PLOT_DIR, "calibration_curves.png")
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_metrics_barchart():
    """Grouped bar chart comparing key metrics across all models."""
    print("\n[Step 6] Plotting Grouped Metrics Bar Chart...")
    files = [
        ("Logistic Regression", "logistic_regression_metrics.csv"),
        ("Decision Tree",       "decision_tree_metrics.csv"),
        ("MLP Neural Net",      "mlp_metrics.csv"),
        ("XGBoost",             "xgboost_metrics.csv"),
        ("LightGBM",            "lightgbm_metrics.csv"),
        ("SVM",                 "svm_metrics.csv"),
        ("Random Forest",       "random_forest_metrics.csv"),
    ]

    records = []
    for model_label, fname in files:
        fpath = os.path.join(RESULT_DIR, fname)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            row = df.iloc[0]
            records.append({
                "Model": model_label,
                "ROC-AUC": row["ROC_AUC"],
                "Recall (Class 1)": row["Recall_Class1"],
                "Precision (Class 1)": row["Precision_Class1"],
                "F1-Score": row["F1_Class1"],
                "Accuracy": row["Accuracy"],
            })

    metrics_df = pd.DataFrame(records)
    melted_df = pd.melt(metrics_df, id_vars=["Model"], var_name="Metric", value_name="Score")

    plt.figure(figsize=(14, 7), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid")
    
    palette = ["#1f77b4", "#ff7f0e", "#9467bd", "#2ca02c", "#17becf", "#d62728", "#8c564b"]
    ax = sns.barplot(data=melted_df, x="Metric", y="Score", hue="Model", palette=palette)
    
    # Annotate values above bars
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f"{height:.2f}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha="center", va="bottom", fontsize=7.5, rotation=0,
                        xytext=(0, 2), textcoords="offset points")

    plt.ylim([0, 1.05])
    plt.xlabel("Evaluation Metric", fontsize=13, labelpad=8)
    plt.ylabel("Score", fontsize=13, labelpad=8)
    plt.title("Comparative Performance Across 7 Models on Key Credit Risk Metrics\nOverall Model Evaluation (N = 45,000)", fontsize=14, fontweight="bold", pad=12)
    plt.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.5)
    plt.tight_layout()

    out_path = os.path.join(PLOT_DIR, "metrics_comparison_barchart.png")
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_confusion_matrices(y_test, predictions):
    """Plot confusion matrices in a dynamic grid (2 rows, 4 cols max)."""
    print("\n[Step 7] Plotting Confusion Matrices...")
    n_models = len(predictions)
    n_cols = 4
    n_rows = (n_models + n_cols - 1) // n_cols  # ceiling division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4.5), dpi=300)
    axes = axes.flatten()

    for idx, (name, data) in enumerate(predictions.items()):
        cm = confusion_matrix(y_test, data["y_pred"])
        ax = axes[idx]
        
        # Annotations formatted with counts and percentages
        labels = [
            [f"TN\n{cm[0,0]:,}\n({cm[0,0]/cm[0].sum():.1%})", f"FP\n{cm[0,1]:,}\n({cm[0,1]/cm[0].sum():.1%})"],
            [f"FN\n{cm[1,0]:,}\n({cm[1,0]/cm[1].sum():.1%})", f"TP\n{cm[1,1]:,}\n({cm[1,1]/cm[1].sum():.1%})"]
        ]
        
        sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", cbar=False, ax=ax,
                    annot_kws={"fontsize": 11, "fontweight": "bold"})
        ax.set_title(f"{name}", fontsize=13, fontweight="bold", pad=8)
        ax.set_xlabel("Predicted Label (0: Non-Default, 1: Default)", fontsize=10.5)
        ax.set_ylabel("Actual Label", fontsize=10.5)
        ax.set_xticklabels(["Class 0", "Class 1"])
        ax.set_yticklabels(["Class 0", "Class 1"])

    # Hide any unused axes
    for idx in range(len(predictions), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Confusion Matrices — Overall Model Performance (N = 45,000)\nHigh Recall Minimizes Costly False Negatives (FN)",
                 fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()

    out_path = os.path.join(PLOT_DIR, "confusion_matrices.png")
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


def run_visualizations():
    os.makedirs(PLOT_DIR, exist_ok=True)
    y_test, predictions = load_models_and_predict()
    
    plot_roc_curves(y_test, predictions)
    plot_precision_recall_curves(y_test, predictions)
    plot_calibration_curves(y_test, predictions)
    plot_metrics_barchart()
    plot_confusion_matrices(y_test, predictions)

    print("\n" + "=" * 65)
    print("PHASE 10A VISUALIZATIONS COMPLETE -- All plots saved to plots/")
    print("=" * 65)


if __name__ == "__main__":
    run_visualizations()
