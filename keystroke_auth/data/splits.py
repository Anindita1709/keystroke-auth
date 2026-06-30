"""
Train / validation / test splitting and impostor sampling.

All split logic lives here so the rest of the codebase stays dataset-agnostic.
"""
import logging

import numpy as np
import pandas as pd

from keystroke_auth.constants import TRAIN_SESSION_END, VAL_SESSION_END

logger = logging.getLogger(__name__)


def session_split(
    df: pd.DataFrame,
    user: str,
    features: list[str],
    train_end: int = TRAIN_SESSION_END,
    val_end: int = VAL_SESSION_END,
) -> dict[str, np.ndarray]:
    """
    Split a user's data chronologically by session index.

    Partitions
    ----------
    sessions 1 – train_end        → "train"
    sessions train_end+1 – val_end → "val"
    sessions val_end+1 – 8        → "test"  (never touched during tuning)

    Why not random split?
        Keystroke timing improves with practice. A random split leaks future
        consistency into training, inflating all reported metrics. The
        session-based protocol matches Killourhy & Maxion (2009).

    Parameters
    ----------
    df        : Full CMU dataframe (all subjects).
    user      : Subject ID string (e.g. "s002").
    features  : Column names to extract.
    train_end : Last session index that belongs to training.
    val_end   : Last session index that belongs to validation.

    Returns
    -------
    dict with keys "train", "val", "test" → np.ndarray of shape (n, n_features).
    """
    u = df[df["subject"] == user]

    splits = {
        "train": u[u["sessionIndex"] <= train_end][features].values,
        "val"  : u[
            (u["sessionIndex"] > train_end) &
            (u["sessionIndex"] <= val_end)
        ][features].values,
        "test" : u[u["sessionIndex"] > val_end][features].values,
    }

    logger.debug(
        "session_split(%s) → train: %d, val: %d, test: %d",
        user, len(splits["train"]), len(splits["val"]), len(splits["test"]),
    )
    return splits


def sample_impostors(
    df: pd.DataFrame,
    genuine_user: str,
    features: list[str],
    n: int = 250,
    random_state: int | None = None,
) -> np.ndarray:
    """
    Draw n timing records from every subject except genuine_user,
    sampled evenly across all other subjects.

    Usage context
    -------------
    One-Class SVM : call this ONLY at evaluation time, never during training.
    Binary SVM    : call this during training to create the negative class.

    Parameters
    ----------
    df            : Full CMU dataframe.
    genuine_user  : The subject whose records must be excluded.
    features      : Column names to extract.
    n             : Total number of impostor rows to return.
    random_state  : Seed for reproducibility.

    Returns
    -------
    np.ndarray of shape (≤n, n_features).
    """
    rng    = np.random.default_rng(random_state)
    others = [u for u in df["subject"].unique() if u != genuine_user]
    per_user = max(1, n // len(others))

    chunks: list[np.ndarray] = []
    for u in others:
        data = df[df["subject"] == u][features].values
        idx  = rng.choice(len(data), min(per_user, len(data)), replace=False)
        chunks.append(data[idx])

    samples = np.vstack(chunks)[:n]
    logger.debug(
        "sample_impostors(genuine=%s, n=%d) → sampled %d rows from %d users",
        genuine_user, n, len(samples), len(others),
    )
    return samples
