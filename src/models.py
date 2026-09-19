"""
=============================================================================
MODEL DEFINITIONS
=============================================================================
Project : GMSC Credit Scoring Research
File    : src/models.py
Purpose : Build each model as a full sklearn Pipeline (preprocessor + model).
          Preprocessing is INSIDE the pipeline so it is refitted on every
          CV fold -- preventing any data leakage.

Models:
  1. Logistic Regression (baseline / traditional statistical model)
  2. Decision Tree (CART)
  3. XGBoost
  4. MLP (Multi-Layer Perceptron)
  5. LightGBM
  6. Support Vector Machine (Calibrated Linear SVM)
  7. Random Forest (Bagging ensemble of decision trees)

Reference: Vakrani et al. (2026) -- adapted to Give Me Some Credit dataset
=============================================================================
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.pipeline         import Pipeline
from sklearn.linear_model     import LogisticRegression
from sklearn.tree             import DecisionTreeClassifier
from sklearn.neural_network   import MLPClassifier
from sklearn.ensemble         import RandomForestClassifier
from xgboost                  import XGBClassifier

from data_preprocessing import build_preprocessing_pipeline
from config import (
    RANDOM_STATE, MLP_EPOCHS, MLP_HIDDEN_1, MLP_HIDDEN_2, MLP_ACTIVATION
)


# =============================================================================
# MODEL 1 -- LOGISTIC REGRESSION
# =============================================================================

def build_logistic_pipeline(C: float = 1.0) -> Pipeline:
    """
    Build a full Pipeline: Preprocessor --> LogisticRegression.

    Preprocessing is INSIDE the pipeline so it refits on each CV fold.

    Parameters
    ----------
    C : float
        Inverse of regularization strength. Larger = less regularization.

    Returns
    -------
    sklearn Pipeline
    """
    preprocessor = build_preprocessing_pipeline()

    model = LogisticRegression(
        C=C,
        class_weight="balanced",   # handles 6.68% minority class
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   model),
    ])

    return pipeline


# =============================================================================
# MODEL 2 -- DECISION TREE (CART)
# =============================================================================

def build_decision_tree_pipeline(
    max_depth: int = 5,
    min_samples_split: int = 20,
    criterion: str = "gini",
) -> Pipeline:
    """
    Build a full Pipeline: Preprocessor --> DecisionTreeClassifier.

    Parameters
    ----------
    max_depth          : int   -- maximum tree depth [3, 10]
    min_samples_split  : int   -- min samples to split a node [5, 50]
    criterion          : str   -- split criterion (gini per methodology)
    """
    preprocessor = build_preprocessing_pipeline()

    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        criterion=criterion,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   model),
    ])

    return pipeline


# =============================================================================
# MODEL 3 -- XGBOOST
# =============================================================================

def build_xgboost_pipeline(
    learning_rate: float = 0.1,
    max_depth: int = 4,
    n_estimators: int = 100,
    scale_pos_weight: float = 1.0,
) -> Pipeline:
    """
    Build a full Pipeline: Preprocessor --> XGBClassifier.

    Parameters
    ----------
    learning_rate     : float  -- step size shrinkage [0.01, 0.3]
    max_depth         : int    -- max tree depth [3, 6]
    n_estimators      : int    -- number of boosting rounds [100, 300]
    scale_pos_weight  : float  -- class imbalance weight (train-data only)
                                  = count(class 0) / count(class 1)
    """
    preprocessor = build_preprocessing_pipeline()

    model = XGBClassifier(
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_estimators=n_estimators,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        verbosity=0,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   model),
    ])

    return pipeline


# =============================================================================
# MODEL 4 -- MLP (Multi-Layer Perceptron)
# =============================================================================

def build_mlp_pipeline(use_class_weight: bool = False) -> Pipeline:
    """
    Build a full Pipeline: Preprocessor --> MLPClassifier.

    Architecture (fixed per methodology):
      Input: 10 features
      Dense 64  -- ReLU
      Dense 32  -- ReLU
      Output    -- sigmoid (binary classification)

    Parameters
    ----------
    use_class_weight : bool
        sklearn MLPClassifier does not natively support class_weight.
        When True, we note this and recommend sample_weight at fit time.
        This flag is used as a reminder/label, not a direct parameter.
    """
    preprocessor = build_preprocessing_pipeline()

    model = MLPClassifier(
        hidden_layer_sizes=(MLP_HIDDEN_1, MLP_HIDDEN_2),  # (64, 32)
        activation=MLP_ACTIVATION,                         # relu
        max_iter=MLP_EPOCHS,                               # 50 epochs
        solver="adam",
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        verbose=False,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   model),
    ])

    label = "with class weight approximation" if use_class_weight else "without class weighting"
    print(f"[build_mlp_pipeline] MLP pipeline built ({label}).")
    print(f"  Architecture : Input(10) -> Dense({MLP_HIDDEN_1}, ReLU) -> Dense({MLP_HIDDEN_2}, ReLU) -> Output(1)")
    print(f"  Max epochs   : {MLP_EPOCHS}")
    print(f"  Note: sklearn MLPClassifier does not support class_weight natively.")
    print(f"        Use sample_weight in pipeline.fit() for imbalance handling.")

    return pipeline, use_class_weight


# =============================================================================
# MODEL 5 -- LIGHTGBM (Light Gradient Boosting Machine)
# =============================================================================

def build_lightgbm_pipeline(
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    n_estimators: int = 200,
    scale_pos_weight: float = 1.0,
) -> Pipeline:
    """
    Build a full Pipeline: Preprocessor --> LGBMClassifier.

    Parameters
    ----------
    learning_rate    : float -- shrinkage rate
    num_leaves       : int   -- max tree leaves (controls tree complexity)
    n_estimators     : int   -- number of boosting trees
    scale_pos_weight : float -- class imbalance ratio
    """
    from lightgbm import LGBMClassifier

    preprocessor = build_preprocessing_pipeline()

    model = LGBMClassifier(
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        n_estimators=n_estimators,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        verbose=-1,
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   model),
    ])

    return pipeline


# =============================================================================
# MODEL 6 -- SUPPORT VECTOR MACHINE (Linear SVM with Calibration)
# =============================================================================

def build_svm_pipeline(C: float = 1.0) -> Pipeline:
    """
    Build a full Pipeline: Preprocessor --> Calibrated LinearSVC.

    CalibratedClassifierCV wraps LinearSVC to produce reliable probabilities
    for ROC-AUC calculation on N=105,000 samples.

    Parameters
    ----------
    C : float -- Regularization parameter
    """
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV

    preprocessor = build_preprocessing_pipeline()

    base_svm = LinearSVC(
        C=C,
        dual="auto",
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    cal_svm = CalibratedClassifierCV(estimator=base_svm, cv=3)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   cal_svm),
    ])

    return pipeline


# =============================================================================
# MODEL 7 -- RANDOM FOREST
# =============================================================================

def build_random_forest_pipeline(
    n_estimators: int = 100,
    max_depth: int = None,
    max_features: str = "sqrt",
    min_samples_split: int = 20,
) -> Pipeline:
    """
    Build a full Pipeline: Preprocessor --> RandomForestClassifier.

    Random Forest is a bagging ensemble of decorrelated decision trees.
    Each tree is trained on a bootstrap sample of the data, and only a
    random subset of features is considered at every split -- this
    reduces variance compared to a single decision tree while preserving
    high recall on the minority class via class_weight='balanced'.

    Parameters
    ----------
    n_estimators       : int  -- number of trees in the forest [100, 300]
    max_depth          : int  -- maximum depth of each tree (None = fully grown)
    max_features       : str  -- features per split ('sqrt' per sklearn default)
    min_samples_split  : int  -- min samples required to split a node [10, 50]
    """
    preprocessor = build_preprocessing_pipeline()

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_split=min_samples_split,
        class_weight="balanced",   # handles 6.68% minority class
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   model),
    ])

    return pipeline

