# Model Training Specifications & Convergence Details
### Project: Give Me Some Credit (GMSC) Scoring Research

This document records the exact training configurations, iterations/epochs, stopping criteria, and convergence properties for all four models evaluated in this research benchmark.

---

## 1. Summary of Training Units Across Models

| Model | Training Unit | Search Grid / Range | Final Selected Setting | Actual Iterations / Epochs | Stopping Mechanism | Final Objective Loss / Score |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LightGBM** | Boosting Rounds (Trees) | `[100, 200]` rounds | `n_est=200, leaves=15, lr=0.03` | **200 boosting rounds** | Full grid optimization | CV ROC-AUC: **0.8654** |
| **XGBoost** | Boosting Rounds (Trees) | `[100, 200, 300]` | `n_estimators = 300` | **300 boosting rounds** | Full grid optimization | CV ROC-AUC: **0.8645** |
| **Decision Tree (CART)** | Tree Depth / Leaves | Depth: `[3 - 10]` | `max_depth = 6` | **59 leaf nodes** | Gini Impurity & min-split | CV ROC-AUC: **0.8437** |
| **MLP Neural Network** | Epochs (Full Dataset Passes) | Max 50 epochs | Architecture: `[64, 32]` | **25 epochs** | Early Stopping (`patience=10`) | Training Loss: **0.4921** |
| **Logistic Regression** | Optimization Iterations | $C \in [0.001 - 10.0]$ | $C = 0.001$ | **1000 max_iter** | L-BFGS convergence | CV ROC-AUC: **0.8227** |
| **Support Vector Machine** | Calibrated Optimization | $C \in [0.01 - 10.0]$ | $C = 0.01$ | **2000 max_iter** | LibLinear & Platt calibration | CV ROC-AUC: **0.8165** |

---

## 2. Deep Dive: Multi-Layer Perceptron (MLP) Neural Network

### Training Parameters
- **Architecture**: `Input(10) -> Dense(64, ReLU) -> Dense(32, ReLU) -> Output(1, Sigmoid)`
- **Optimizer**: Adam ($\beta_1 = 0.9, \beta_2 = 0.999$)
- **Initial Learning Rate**: `learning_rate_init = 0.001`
- **L2 Regularization ($\alpha$)**: `0.001`
- **Configured Maximum Epochs**: 50 (`max_iter=50`)
- **Loss Function**: Binary Cross-Entropy with sample weights ($w_{\text{default}} = 13.96, w_{\text{non-default}} = 1.0$)

### Convergence Dynamics
- **Actual Epochs Completed**: **25 epochs**
- **Early Stopping Trigger**: Scikit-Learn's `MLPClassifier` monitored a 10% stratified validation split (`validation_fraction=0.10`).
- Because validation loss did not improve by more than $10^{-4}$ for **10 consecutive epochs** (`n_iter_no_change=10`), training halted at **Epoch 25** to guarantee generalization and prevent weight overfitting.
- **Final Cross-Entropy Loss**: `0.4921`

---

## 3. Deep Dive: XGBoost (Extreme Gradient Boosted Trees)

### Training Parameters
- **Algorithm**: Tree-based gradient boosting (`hist` tree method)
- **Base Tree Depth**: `max_depth = 3` (shallow trees designed to prevent overfitting on noisy retail credit records)
- **Learning Rate / Shrinkage ($\eta$)**: `0.05` (moderate step size ensuring smooth additive corrections)
- **Class Imbalance Scaling**: `scale_pos_weight = 13.96` ($97,982 / 7,018$ negative-to-positive ratio computed strictly on training data)
- **Evaluation Metric**: Area Under the ROC Curve (`auc`)

### Sequential Boosting Rounds
- **Total Boosting Rounds Trained**: **300 rounds (300 sequential decision trees)**
- **How Boosting Works**: Unlike neural network epochs where the same weights are updated in-place, XGBoost adds tree $T_t$ sequentially at round $t \in [1, 300]$:
  $$\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + \eta \cdot f_t(x_i)$$
- Each of the 300 trees directly predicts the pseudo-residuals (negative gradient of the log-loss) from all prior $t-1$ trees.
- **Ensemble Tree Count**: Exactly 300 individual trees are stored inside [`models/xgboost.joblib`](file:///c:/Users/SUMITRANAND%20SHARMA/OneDrive/Desktop/ML_PRO/credit-scoring-project/models/xgboost.joblib).

---

## 4. Deep Dive: Decision Tree (CART)

- **Criterion**: Gini Impurity (fixed per methodology)
- **Max Depth**: `max_depth = 6`
- **Min Samples to Split**: `min_samples_split = 50`
- **Actual Tree Depth Achieved**: 6 levels
- **Total Terminal Leaves**: 59 leaves
- **Class Balancing**: `class_weight = "balanced"`

---

## 5. Deep Dive: Logistic Regression (Baseline)

- **Regularization Type**: L2 (Ridge penalty)
- **Optimal Regularization Strength**: $C = 0.001$ (strong regularization selected to prevent noisy coefficients on collinear delinquency features)
- **Solver**: L-BFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno quasi-Newton algorithm)
- **Tolerance**: $10^{-4}$
- **Class Weighting**: `class_weight = "balanced"`
