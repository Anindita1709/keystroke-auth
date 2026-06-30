"""
Shared pytest fixtures.

We generate a synthetic dataset that mimics the CMU Keystroke Dynamics CSV
schema (subject, sessionIndex, rep, + 31 timing columns) so the full test
suite runs offline and in well under a second — no network access to
cs.cmu.edu required.

Each synthetic user is given a distinct mean/std per feature so that
genuine-vs-impostor separation is realistic enough to exercise EER, ROC,
and PCA code paths meaningfully.
"""
import numpy as np
import pandas as pd
import pytest

from keystroke_auth.constants import RAW_FEATURES


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """
    Build a small synthetic dataset: 6 users x 8 sessions x 5 reps = 240 rows.

    H.*  and DD.* columns are drawn from positive distributions (hold times
    and key-to-key flight times are always >= 0 in real keystroke data).
    UD.* columns are allowed to go negative, matching real key-overlap
    behaviour during fast typing.
    """
    rng = np.random.default_rng(42)
    n_users, n_sessions, n_reps = 6, 8, 5

    rows = []
    for u in range(n_users):
        subject = f"s{u:03d}"
        # Each user gets a distinct "typing signature"
        user_mean = rng.uniform(0.05, 0.25, size=len(RAW_FEATURES))
        user_std  = rng.uniform(0.01, 0.05, size=len(RAW_FEATURES))

        for session in range(1, n_sessions + 1):
            for rep in range(1, n_reps + 1):
                values = rng.normal(user_mean, user_std)
                # UD columns may legitimately be negative; H/DD must be >= 0
                for i, col in enumerate(RAW_FEATURES):
                    if col.startswith("H.") or col.startswith("DD."):
                        values[i] = abs(values[i])
                row = dict(zip(RAW_FEATURES, values))
                row["subject"]      = subject
                row["sessionIndex"] = session
                row["rep"]          = rep
                rows.append(row)

    df = pd.DataFrame(rows)
    # Match the CMU CSV's column ordering: subject, sessionIndex, rep, features...
    return df[["subject", "sessionIndex", "rep"] + RAW_FEATURES]


@pytest.fixture
def synthetic_user(synthetic_df) -> str:
    """The first subject ID in the synthetic dataset."""
    return synthetic_df["subject"].unique()[0]
