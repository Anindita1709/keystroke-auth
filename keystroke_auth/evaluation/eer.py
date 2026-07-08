"""
Equal Error Rate (EER) computation.
"""
import logging
from typing import Any

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import auc, roc_curve

logger = logging.getLogger(__name__)

def compute_eer(
    scores_genuine: np.ndarray,
    scores_impostor: np.ndarray,
) -> dict[str, Any]:
    """
    FAR  (False Acceptance Rate) = impostors accepted  = FPR on the ROC curve
    FRR  (False Rejection Rate)  = genuine users rejected = 1 − TPR
    EER  = operating point where FAR == FRR
    Lower EER -> better system.
    """
    scores = np.concatenate([scores_genuine, scores_impostor])
    labels = np.concatenate([
        np.ones(len(scores_genuine)),
        np.zeros(len(scores_impostor)),
    ])

    fpr, tpr, thresholds = roc_curve(labels, scores)
    frr       = 1.0 - tpr
    auc_score = auc(fpr, tpr)

    try:
        # Numerically find the crossing point FAR = FRR via root-finding.
        # Using brentq (bracketed Brent's method) on the interpolated curve
        eer       = brentq(lambda x: float(interp1d(fpr, frr)(x)) - x, 0.0, 1.0)
        idx       = int(np.argmin(np.abs(fpr - eer)))
        eer_thresh = float(thresholds[idx])
    except (ValueError, IndexError):
        # Curves may not cross if one class is perfectly separated
        eer        = np.nan
        eer_thresh = np.nan
        logger.warning(
            "EER undefined: FAR–FRR curves do not intersect. "
            "Check score distributions for perfect separation."
        )

    logger.debug("EER=%.4f  AUC=%.4f", eer, auc_score)

    return {
        "eer"          : eer,
        "eer_threshold": eer_thresh,
        "auc"          : auc_score,
        "fpr"          : fpr,
        "tpr"          : tpr,
        "frr"          : frr,
        "thresholds"   : thresholds,
    }
