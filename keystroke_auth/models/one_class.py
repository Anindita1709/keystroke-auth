"""
Model is trained exclusively on a genuine user's typing samples and 
learns a decision boundary around them.At inference, samples outside 
the boundary are anomalies (impostors).No impostor data is needed during training.
"""
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from keystroke_auth.data.splits import sample_impostors, session_split
from keystroke_auth.evaluation.eer import compute_eer

logger = logging.getLogger(__name__)

def train_one_class_svm(
    df: pd.DataFrame,
    user: str,
    features: list[str],
    nu: float = 0.1,
    kernel: str = "rbf",
    gamma: str | float = "scale",
) -> dict[str, Any]:
    
    splits = session_split(df, user, features)

    # Fit scaler on TRAINING SET ONLY — never expose val/test statistics
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(splits["train"])
    X_val   = scaler.transform(splits["val"])
    X_test  = scaler.transform(splits["test"])

    # Impostor samples are used ONLY at evaluation time, never during training
    X_imp_val  = scaler.transform(sample_impostors(df, user, features, n=200))
    X_imp_test = scaler.transform(sample_impostors(df, user, features, n=100))

    model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    model.fit(X_train)

    n_sv = model.support_vectors_.shape[0]
    logger.debug("One-Class SVM [%s] — nu=%.3f, gamma=%s, n_sv=%d", user, nu, gamma, n_sv)

    return {
        "model"        : model,
        "scaler"       : scaler,
        "n_sv"         : n_sv,
        "val_genuine"  : model.decision_function(X_val),
        "val_impostor" : model.decision_function(X_imp_val),
        "test_genuine" : model.decision_function(X_test),
        "test_impostor": model.decision_function(X_imp_test),
    }


def tune_one_class_svm(
    df: pd.DataFrame,
    user: str,
    features: list[str],
    nu_grid: list[float] | None = None,
    gamma_grid: list[str | float] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Grid search over nu × gamma using VALIDATION EER as the objective.

    Why not GridSearchCV / cross-validation?
        One-Class SVM trains on a single class, so k-fold CV cannot produce
        meaningful multi-class splits. Validation-set EER is the only valid
        tuning signal.

    """
    nu_grid    = nu_grid    or [0.01, 0.05, 0.1, 0.15, 0.2, 0.3]
    gamma_grid = gamma_grid or ["scale", "auto", 0.01, 0.05, 0.1]

    splits = session_split(df, user, features)
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(splits["train"])
    X_val   = scaler.transform(splits["val"])
    X_imp   = scaler.transform(sample_impostors(df, user, features, n=200))

    best_eer    = float("inf")
    best_params: dict[str, Any] = {"nu": 0.1, "gamma": "scale"}
    records: list[dict] = []

    for nu in nu_grid:
        for gamma in gamma_grid:
            try:
                m   = OneClassSVM(nu=nu, kernel="rbf", gamma=gamma).fit(X_train)
                met = compute_eer(
                    m.decision_function(X_val),
                    m.decision_function(X_imp),
                )
                records.append({"nu": nu, "gamma": str(gamma), **{
                    k: v for k, v in met.items()
                    if k in ("eer", "auc")   # keep DataFrame lightweight
                }})

                if not np.isnan(met["eer"]) and met["eer"] < best_eer:
                    best_eer    = met["eer"]
                    best_params = {"nu": nu, "gamma": gamma}

            except Exception as exc:
                logger.debug("Skip nu=%.2f gamma=%s: %s", nu, gamma, exc)

    logger.info(
        "Tuning [%s] — best nu=%s, gamma=%s, val EER=%.2f%%",
        user, best_params["nu"], best_params["gamma"], best_eer * 100,
    )
    return best_params, pd.DataFrame(records)
