"""Model evaluation plots: hyperparameter tuning, score distributions, ROC/DET curves."""
import logging
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keystroke_auth.visualization._style import save_and_show, BLUE, ORANGE, RED

logger = logging.getLogger(__name__)


def plot_tuning_heatmap(tuning_df: pd.DataFrame, show: bool = True) -> None:
    """
    Heatmap of validation EER across the nu × gamma grid search.
    """
    pivot = tuning_df.pivot_table(index="nu", columns="gamma",
                                   values="eer", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values * 100, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=40)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xlabel("gamma")
    ax.set_ylabel("nu")
    ax.set_title("One-Class SVM tuning — Validation EER (%)\nGreen = low EER = better authentication")
    fig.colorbar(im, ax=ax, label="EER (%)")
    save_and_show(fig, "tuning_heatmap.png", show=show)


def plot_score_distributions(
    scores_gen: np.ndarray,
    scores_imp: np.ndarray,
    eer: float,
    user_id: str,
    model_name: str,
    show: bool = True,
) -> None:
    """
    genuine vs. impostor decision-score histograms.
    The overlap region between the two histograms IS the authentication error
    
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(scores_gen, bins=30, alpha=0.65, color=BLUE,
            label=f"Genuine user  (n={len(scores_gen)})", density=True)
    ax.hist(scores_imp, bins=30, alpha=0.65, color=ORANGE,
            label=f"Impostors     (n={len(scores_imp)})", density=True)

    if not np.isnan(eer):
        threshold = np.percentile(scores_gen, eer * 100)
        ax.axvline(threshold, color=RED, linestyle="--", linewidth=1.8,
                   label=f"EER threshold ≈ {eer * 100:.1f}%")

    ax.set_xlabel("SVM decision score (higher → more genuine)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"Score distributions — {user_id}  [{model_name}]\nOverlap region = authentication errors")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    save_and_show(fig, f"score_dist_{user_id}.png", show=show)


def plot_det_roc(
    oc_metrics: dict[str, Any],
    bin_metrics: dict[str, Any] | None,
    user_id: str,
    show: bool = True,
) -> None:
    """
    ROC and DET (Detection Error Tradeoff) curves comparing
    One-Class SVM against Binary SVM.

    The DET curve is the standard chart in biometrics — it plots FAR vs FRR
    and marks the EER as the point closest to the diagonal.
    oc_metrics  : Output of compute_eer() for the One-Class SVM.
    bin_metrics : Output of compute_eer()-equivalent dict for Binary SVM,
                  or None to plot One-Class SVM alone.
    
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── ROC ──────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(oc_metrics["fpr"] * 100, oc_metrics["tpr"] * 100, color=BLUE,
            lw=2, label=f"One-Class SVM  AUC={oc_metrics['auc']:.3f}")
    if bin_metrics:
        ax.plot(bin_metrics["fpr"] * 100, bin_metrics["tpr"] * 100, color=ORANGE,
                lw=2, label=f"Binary SVM     AUC={bin_metrics['auc']:.3f}")
    ax.plot([0, 100], [0, 100], "k--", alpha=0.35, label="Random baseline")
    ax.set_xlabel("False Acceptance Rate (%)")
    ax.set_ylabel("True Acceptance Rate (%)")
    ax.set_title(f"ROC — {user_id}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # ── DET ──────────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(oc_metrics["fpr"] * 100, oc_metrics["frr"] * 100, color=BLUE,
            lw=2, label=f"One-Class SVM  EER={oc_metrics['eer'] * 100:.1f}%")
    if not np.isnan(oc_metrics["eer"]):
        e = oc_metrics["eer"] * 100
        ax.plot(e, e, "o", color=BLUE, ms=9, zorder=5)

    if bin_metrics:
        ax.plot(bin_metrics["fpr"] * 100, bin_metrics["frr"] * 100, color=ORANGE,
                lw=2, label=f"Binary SVM     EER={bin_metrics['eer'] * 100:.1f}%")
        if not np.isnan(bin_metrics["eer"]):
            e = bin_metrics["eer"] * 100
            ax.plot(e, e, "o", color=ORANGE, ms=9, zorder=5)

    ax.plot([0, 50], [0, 50], "k--", alpha=0.35)
    ax.set_xlabel("False Acceptance Rate (%)")
    ax.set_ylabel("False Rejection Rate (%)")
    ax.set_title(f"DET Curve — {user_id}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    save_and_show(fig, f"det_roc_{user_id}.png", show=show)
