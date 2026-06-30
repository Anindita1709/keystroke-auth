"""System-wide evaluation plots: EER distribution across all users."""
import logging

import pandas as pd
import matplotlib.pyplot as plt

from keystroke_auth.visualization._style import save_and_show, BLUE, RED

logger = logging.getLogger(__name__)


def plot_system_eer(results_df: pd.DataFrame, show: bool = True) -> None:
    """
    Plot the EER distribution and a sorted per-user ranking across all
    evaluated subjects.

    Left panel  : histogram of EER values with the mean marked — shows
                  whether errors are tightly clustered or long-tailed.
    Right panel : sorted horizontal bar chart — identifies which specific
                  users are hardest to authenticate (useful for a
                  "what would you investigate next" interview follow-up).

    Parameters
    ----------
    results_df : Output of evaluate_all_users() — must have an 'eer' column.
    show       : Whether to call plt.show() after saving.
    """
    mean_eer = results_df["eer"].mean() * 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Histogram ────────────────────────────────────────────────────────
    axes[0].hist(results_df["eer"] * 100, bins=18, color=BLUE,
                 edgecolor="white", linewidth=0.5)
    axes[0].axvline(mean_eer, color=RED, linestyle="--", lw=1.8,
                    label=f"Mean EER: {mean_eer:.1f}%")
    axes[0].set_xlabel("EER (%)")
    axes[0].set_ylabel("# users")
    axes[0].set_title(f"EER distribution — all {len(results_df)} users")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    # ── Sorted per-user bar chart ────────────────────────────────────────
    sorted_df = results_df.sort_values("eer")
    axes[1].barh(range(len(sorted_df)), sorted_df["eer"] * 100, color=BLUE, alpha=0.7)
    axes[1].axvline(mean_eer, color=RED, linestyle="--", alpha=0.8)
    axes[1].set_xlabel("EER (%)")
    axes[1].set_title("Per-user EER (sorted low → high)")
    axes[1].set_yticks([])
    axes[1].grid(True, alpha=0.2, axis="x")

    save_and_show(fig, "system_eer.png", show=show)
