"""
Trade-off vs One-Class SVM
──────────────────────────
Advantage  : Usually achieves lower EER because it directly models the
             impostor distribution.
Drawbacks  : Requires impostor data at enrollment (less realistic);
             sensitive to which impostors were observed during training
             (the open-set problem).
"""
import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from keystroke_auth.data.splits import sample_impostors, session_split

logger = logging.getLogger(__name__)

_DEFAULT_PARAM_GRID: dict[str, list] = {
    "C"     : [0.1, 1, 10, 100],
    "gamma" : ["scale", 0.01, 0.1],
    "kernel": ["rbf", "linear"],
}

def train_binary_svm(
    df: pd.DataFrame,
    user: str,
    features: list[str],
    param_grid: dict | None = None,
    cv_folds: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Training set  : genuine user (y=1) + balanced impostor sample (y=0)
    Tuning        : GridSearchCV with StratifiedKFold, scored by ROC-AUC
    class_weight  : 'balanced' to handle any residual label imbalance

    """
    param_grid = param_grid or _DEFAULT_PARAM_GRID
    splits = session_split(df, user, features)

    # ── Build training set ──────────────────────────────────────────────────
    X_gen  = splits["train"]
    X_imp  = sample_impostors(df, user, features, n=len(X_gen),
                               random_state=random_state)

    X_train = np.vstack([X_gen, X_imp])
    y_train = np.concatenate([np.ones(len(X_gen)), np.zeros(len(X_imp))])

    rng  = np.random.default_rng(random_state)
    perm = rng.permutation(len(X_train))
    X_train, y_train = X_train[perm], y_train[perm]

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # ── Grid search ────────────────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                         random_state=random_state)
    gs = GridSearchCV(
        SVC(probability=True, class_weight="balanced"),
        param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,
    )
    gs.fit(X_scaled, y_train)

    logger.info(
        "Binary SVM [%s] — best params: %s  CV AUC: %.4f",
        user, gs.best_params_, gs.best_score_,
    )

    # ── Test set evaluation ────────────────────────────────────────────────
    X_test_gen = scaler.transform(splits["test"])
    X_test_imp = scaler.transform(
        sample_impostors(df, user, features, n=100,
                         random_state=random_state + 1)
    )
    X_test = np.vstack([X_test_gen, X_test_imp])
    y_test = np.concatenate([
        np.ones(len(X_test_gen)),
        np.zeros(len(X_test_imp)),
    ])

    probs       = gs.best_estimator_.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    frr         = 1.0 - tpr

    try:
        eer = brentq(lambda x: float(interp1d(fpr, frr)(x)) - x, 0.0, 1.0)
    except ValueError:
        eer = np.nan
        logger.warning("EER undefined for binary SVM [%s]", user)

    return {
        "model"      : gs.best_estimator_,
        "scaler"     : scaler,
        "best_params": gs.best_params_,
        "cv_auc"     : gs.best_score_,
        "eer"        : eer,
        "auc"        : auc(fpr, tpr),
        "fpr"        : fpr,
        "tpr"        : tpr,
        "frr"        : frr,
    }
