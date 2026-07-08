"""
run  One-Class SVM authentication for every user as the genuine
subject and aggregate results into a single performance summary.
"""
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from keystroke_auth.evaluation.eer import compute_eer
from keystroke_auth.models.one_class import train_one_class_svm

logger = logging.getLogger(__name__)


def evaluate_all_users(
    df: pd.DataFrame,
    features: list[str],
    nu: float = 0.1,
    gamma: str | float = "scale",
    save_path: Path | str | None = "system_results.csv",
) -> pd.DataFrame:

    subjects = df["subject"].unique()
    records: list[dict[str, Any]] = []

    for i, user in enumerate(subjects, start=1):
        try:
            res = train_one_class_svm(df, user, features, nu=nu, gamma=gamma)
            met = compute_eer(res["test_genuine"], res["test_impostor"])
            records.append({
                "user": user,
                "eer" : met["eer"],
                "auc" : met["auc"],
                "n_sv": res["n_sv"],
            })
        except Exception as exc:
            logger.warning("Skipping user %s: %s", user, exc)

        if i % 10 == 0 or i == len(subjects):
            logger.info("Progress: %d/%d users evaluated", i, len(subjects))

    results_df = pd.DataFrame(records)
    _log_summary(results_df)

    if save_path is not None:
        results_df.to_csv(save_path, index=False)
        logger.info("Saved per-user results → %s", save_path)

    return results_df

def _log_summary(results_df: pd.DataFrame) -> None:
   
    if results_df.empty:
        logger.warning("No users were successfully evaluated.")
        return

    best  = results_df.loc[results_df["eer"].idxmin()]
    worst = results_df.loc[results_df["eer"].idxmax()]

    sep = "=" * 55
    logger.info(sep)
    logger.info("SYSTEM-WIDE RESULTS — One-Class SVM (nu-SVM)")
    logger.info(sep)
    logger.info("Mean EER   : %.2f%%", results_df["eer"].mean() * 100)
    logger.info("Median EER : %.2f%%", results_df["eer"].median() * 100)
    logger.info("Std EER    : %.2f%%", results_df["eer"].std() * 100)
    logger.info("Best EER   : %.2f%%  (user: %s)", best["eer"] * 100, best["user"])
    logger.info("Worst EER  : %.2f%%  (user: %s)", worst["eer"] * 100, worst["user"])
    logger.info(sep)
