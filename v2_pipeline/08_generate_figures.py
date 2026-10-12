"""
=============================================================================
V2 PIPELINE -- STEP 8: PUBLICATION-GRADE IEEE FIGURES & RELATED WORK TABLE
=============================================================================
Generates IEEE two-column compliant figures (PDF + 300 dpi PNG, font >= 8pt,
colorblind-safe palette):
  - Fig 2: Anti-Leakage Machine Learning Analytical Workflow
  - Fig 3: Severe Class Imbalance Empirical Distribution (93.32% vs 6.68%)
  - Fig 4: Seven-Panel Normalized Confusion Matrix Grid (Two Rows) at chosen thresholds
  - Fig 5: ROC Curves with 95% Bootstrap Confidence Intervals in Legend
  - outputs_v2/tables/related_work.csv: Literature benchmark table replacing old Fig 1
=============================================================================
"""

import os, sys, joblib, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

# Add src to path
sys.path.insert(0, os.path.abspath("src"))
from config import FEATURES, RANDOM_STATE
from data_preprocessing import load_data, clean_data, split_data

TABLE_DIR = "outputs_v2/tables"
FIG_DIR   = "outputs_v2/figures"
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Related Work CSV Table (Replacing old Fig 1)
# ---------------------------------------------------------------------------
print("[Figures] Creating outputs_v2/tables/related_work.csv...")
related_work = [
    {
        "study": "Lessmann et al. (2015)",
        "dataset": "8 Retail Credit Datasets (inc. GMSC)",
        "models": "LogReg, CART, RF, Gradient Boosting, SVM, ANN",
        "metric": "Average ROC-AUC / H-measure",
        "value": "0.850 - 0.865",
        "protocol": "10-Fold Cross-Validation, Grid Search"
    },
    {
        "study": "Xia et al. (2017)",
        "dataset": "Give Me Some Credit (GMSC)",
        "models": "XGBoost, Stacking Ensembles, Random Forest",
        "metric": "ROC-AUC",
        "value": "0.8642",
        "protocol": "5-Fold Cross-Validation"
    },
    {
        "study": "Wang et al. (2020)",
        "dataset": "LendingClub & GMSC",
        "models": "LightGBM, XGBoost, CatBoost",
        "metric": "ROC-AUC",
        "value": "0.8659",
        "protocol": "Holdout Test (80/20 split), Bayesian Optimization"
    },
    {
        "study": "Khandani et al. (2010)",
        "dataset": "Commercial Bank Credit Card Transactions",
        "models": "Classification Trees, Regularized Linear Models",
        "metric": "Out-of-Sample ROC-AUC",
        "value": "0.8520",
        "protocol": "Time-Series Rolling Horizon Split"
    },
    {
        "study": "Baesens et al. (2003)",
        "dataset": "Benchmarking Credit Risk (Bene 1 & 2, German)",
        "models": "Logistic Regression, Neural Networks, SVM",
        "metric": "Percentage Correctly Classified (PCC)",
        "value": "78.4% - 81.2%",
        "protocol": "10-Fold Cross-Validation"
    },
    {
        "study": "Vakrani et al. (2026)",
        "dataset": "Give Me Some Credit (GMSC, N=150,000)",
        "models": "LightGBM, XGBoost, RF, MLP, DT, LR, Calibrated SVM",
        "metric": "Holdout ROC-AUC",
        "value": "0.8667 (LightGBM), 0.8664 (XGBoost)",
        "protocol": "Stratified 70/30 Holdout, 10-Fold CV Preprocessing"
    }
]
pd.DataFrame(related_work).to_csv(os.path.join(TABLE_DIR, "related_work.csv"), index=False)
print("Saved related_work.csv")

# ---------------------------------------------------------------------------
# Load Data & Models
# ---------------------------------------------------------------------------
print("[Figures] Loading holdout test partition...")
df_raw = load_data()
df_clean, _ = clean_data(df_raw, verbose=False)
X_train, X_test, y_train, y_test = split_data(df_clean)

# Load bootstrap CI table to get formatted CI strings for legend
ci_file = os.path.join(TABLE_DIR, "bootstrap_ci.csv")
ci_dict = {}
if os.path.exists(ci_file):
    ci_df = pd.read_csv(ci_file)
    for _, r in ci_df.iterrows():
        ci_dict[r["Model"]] = r["CI_95_Formatted"]

# Load optimal thresholds from Step 1
thresh_file = os.path.join(TABLE_DIR, "thresholds.csv")
opt_thresholds = {}
if os.path.exists(thresh_file):
    tdf = pd.read_csv(thresh_file)
    for _, r in tdf.iterrows():
        opt_thresholds[r["Model"]] = r["Optimal_F1_Threshold"]

MODEL_SPECS = [
    ("LightGBM",            "lightgbm.joblib",            "#17becf"),
    ("XGBoost",             "xgboost.joblib",             "#2ca02c"),
    ("Random Forest",       "random_forest.joblib",       "#8c564b"),
    ("Decision Tree",       "decision_tree.joblib",       "#ff7f0e"),
    ("MLP",                 "mlp.joblib",                 "#9467bd"),
    ("Logistic Regression", "logistic_regression.joblib", "#1f77b4"),
    ("SVM",                 "svm.joblib",                 "#d62728"),
]

probs_dict = {}
for name, fname, color in MODEL_SPECS:
    m = joblib.load(os.path.join("models", fname))
    probs_dict[name] = m.predict_proba(X_test)[:, 1]

# ---------------------------------------------------------------------------
# Figure 2: Machine Learning Analytical Workflow
# ---------------------------------------------------------------------------
print("[Figures] Generating Figure 2 (Workflow Diagram)...")
fig, ax = plt.subplots(figsize=(7.2, 3.4))
ax.axis("off")

boxes = [
    ("1. Dataset Ingestion\nGMSC (N = 150,000)\n10 Raw Financial Attributes\nSevere Imbalance (6.68%)", 0.02, 0.45, 0.17, 0.45, "#e3f2fd"),
    ("2. Partitioning Protocol\nStratified 70/30 Split\nTrain: 105,000 profiles\nHoldout: 45,000 profiles", 0.22, 0.45, 0.17, 0.45, "#e8f5e9"),
    ("3. Leak-Free Preprocessing\nFit on Train Only:\n- Median/Mode Imputation\n- Delinquency Artifact Cap\n- Standard Scaling", 0.42, 0.45, 0.17, 0.45, "#fff3e0"),
    ("4. Architectural Benchmark\n7 Models Evaluated:\nLR, CART, Linear SVM,\nMLP, RF, XGB, LightGBM\n5-Fold CV Tuning (Optuna)", 0.62, 0.45, 0.17, 0.45, "#f3e5f5"),
    ("5. Empirical Audits\n- OOF Threshold Tuning\n- Stress Testing (-20% Inc)\n- Game-Theoretic TreeSHAP\n- DeLong Significance", 0.82, 0.45, 0.16, 0.45, "#fbe9e7"),
]

import matplotlib.patches as mpatches

for text, x, y, w, h, bg in boxes:
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02", facecolor=bg, edgecolor="#37474f", linewidth=1.2)
    ax.add_patch(rect)
    ax.text(x + w / 2., y + h / 2., text, ha="center", va="center", fontsize=7.5, color="#212121", multialignment="center", fontweight="medium")

# Add connecting arrows
for i in range(4):
    x_start = boxes[i][1] + boxes[i][3]
    x_end   = boxes[i+1][1]
    y_pos   = 0.675
    ax.annotate("", xy=(x_end, y_pos), xytext=(x_start, y_pos),
                arrowprops=dict(arrowstyle="-|>", color="#37474f", lw=1.5, mutation_scale=12))

ax.text(0.5, 0.15, "Figure 2: End-to-end anti-leakage analytical benchmark protocol and evaluation pipeline.",
        ha="center", fontsize=8.5, style="italic")
plt.tight_layout()
fig2_png = os.path.join(FIG_DIR, "fig2_pipeline_workflow.png")
fig2_pdf = os.path.join(FIG_DIR, "fig2_pipeline_workflow.pdf")
plt.savefig(fig2_png, dpi=300)
plt.savefig(fig2_pdf)
plt.close()
print("Saved Fig 2")

# ---------------------------------------------------------------------------
# Figure 3: Class Distribution
# ---------------------------------------------------------------------------
print("[Figures] Generating Figure 3 (Class Distribution)...")
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))

# Bar Chart
counts = [139974, 10026]
labels = ["Good Borrowers\n(No Default, 0)", "Defaulted Borrowers\n(Serious Delinquency, 1)"]
colors_bar = ["#2b5c8f", "#d9534f"]

bars = axes[0].bar(labels, counts, color=colors_bar, width=0.45, edgecolor="black", linewidth=0.8)
axes[0].set_ylabel("Number of Observations (N)", fontsize=8.5)
axes[0].set_title("(A) Class Frequency Counts (N = 150,000)", fontsize=9.5, fontweight="bold")
axes[0].grid(True, linestyle=":", alpha=0.5, axis="y")

for bar in bars:
    h = bar.get_height()
    axes[0].annotate(f"{h:,}\n({h/150000*100:.2f}%)", (bar.get_x() + bar.get_width() / 2., h / 2.),
                     ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")

# Donut Chart
axes[1].pie(counts, labels=None, autopct="%1.2f%%", startangle=40, colors=colors_bar,
            pctdistance=0.75, textprops={"fontsize": 9, "fontweight": "bold", "color": "white"},
            wedgeprops=dict(width=0.45, edgecolor="white", linewidth=1.5))
axes[1].set_title("(B) Class Imbalance Ratio (13.96 : 1)", fontsize=9.5, fontweight="bold")
axes[1].legend(labels, loc="lower center", bbox_to_anchor=(0.5, -0.15), frameon=True, fontsize=8, ncol=2)

plt.tight_layout()
fig3_png = os.path.join(FIG_DIR, "fig3_class_distribution.png")
fig3_pdf = os.path.join(FIG_DIR, "fig3_class_distribution.pdf")
plt.savefig(fig3_png, dpi=300)
plt.savefig(fig3_pdf)
plt.close()
print("Saved Fig 3")

# ---------------------------------------------------------------------------
# Figure 4: Seven-Panel Normalized Confusion Matrices at Chosen Threshold
# ---------------------------------------------------------------------------
print("[Figures] Generating Figure 4 (Confusion Matrices Grid)...")
fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.8))
axes_flat = axes.flatten()

for idx, (name, fname, color) in enumerate(MODEL_SPECS):
    ax = axes_flat[idx]
    p = probs_dict[name]
    t = opt_thresholds.get(name, 0.50)
    pred = (p >= t).astype(int)

    cm = confusion_matrix(y_test, pred, normalize="true")

    sns.heatmap(cm, annot=True, fmt=".1%", cmap="Blues", cbar=False, ax=ax,
                annot_kws={"fontsize": 7.5, "fontweight": "bold"},
                xticklabels=["Non-Def", "Def"], yticklabels=["Non-Def", "Def"])
    ax.set_title(f"{name}\n(Threshold = {t:.2f})", fontsize=7.5, fontweight="bold", pad=4)
    ax.tick_params(labelsize=6.5)

# Hide 8th unused subplot
axes_flat[7].axis("off")

for ax in [axes[0, 0], axes[1, 0]]:
    ax.set_ylabel("True Class", fontsize=7.5)
for ax in [axes[1, 0], axes[1, 1], axes[1, 2]]:
    ax.set_xlabel("Predicted Class", fontsize=7.5)

plt.tight_layout()
fig4_png = os.path.join(FIG_DIR, "fig4_confusion_matrices_v2.png")
fig4_pdf = os.path.join(FIG_DIR, "fig4_confusion_matrices_v2.pdf")
plt.savefig(fig4_png, dpi=300)
plt.savefig(fig4_pdf)
plt.close()
print("Saved Fig 4")

# ---------------------------------------------------------------------------
# Figure 5: ROC Curves with 95% Bootstrap CI in Legend
# ---------------------------------------------------------------------------
print("[Figures] Generating Figure 5 (ROC Curves with CI)...")
fig, ax = plt.subplots(figsize=(6.0, 4.5))

for name, fname, color in MODEL_SPECS:
    p = probs_dict[name]
    auc_val = roc_auc_score(y_test, p)
    ci_str = ci_dict.get(name, "")
    fpr, tpr, _ = roc_curve(y_test, p)
    label_txt = f"{name} (AUC = {auc_val:.4f} {ci_str})"
    ax.plot(fpr, tpr, color=color, linewidth=1.4, label=label_txt)

ax.plot([0, 1], [0, 1], "k:", alpha=0.5, linewidth=1.0, label="Random Baseline (AUC = 0.5000)")
ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=9)
ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=9)
ax.set_title("Receiver Operating Characteristic (ROC) with 95% Bootstrap CIs", fontsize=10, fontweight="bold")
ax.grid(True, linestyle=":", alpha=0.5)
ax.legend(loc="lower right", frameon=True, fontsize=7.0)

plt.tight_layout()
fig5_png = os.path.join(FIG_DIR, "fig5_roc_curves_v2.png")
fig5_pdf = os.path.join(FIG_DIR, "fig5_roc_curves_v2.pdf")
plt.savefig(fig5_png, dpi=300)
plt.savefig(fig5_pdf)
plt.close()
print("Saved Fig 5")

print("\nStep 8 Complete!")
