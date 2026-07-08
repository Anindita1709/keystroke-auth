"""
Train / validation / test splitting and impostor sampling.
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
