"""Tests for keystroke_auth.data.splits."""
import numpy as np

from keystroke_auth.constants import RAW_FEATURES
from keystroke_auth.data.splits import sample_impostors, session_split


def test_session_split_partitions_are_disjoint(synthetic_df, synthetic_user):
    splits = session_split(synthetic_df, synthetic_user, RAW_FEATURES,
                           train_end=5, val_end=7)

    # synthetic fixture: 5 reps/session -> 5 train sessions = 25 rows,
    # 2 val sessions = 10 rows, 1 test session = 5 rows
    assert splits["train"].shape == (25, len(RAW_FEATURES))
    assert splits["val"].shape   == (10, len(RAW_FEATURES))
    assert splits["test"].shape  == (5,  len(RAW_FEATURES))


def test_session_split_respects_custom_boundaries(synthetic_df, synthetic_user):
    splits = session_split(synthetic_df, synthetic_user, RAW_FEATURES,
                           train_end=6, val_end=7)
    assert splits["train"].shape[0] == 30  # 6 sessions x 5 reps
    assert splits["val"].shape[0]   == 5   # 1 session x 5 reps
    assert splits["test"].shape[0]  == 5   # 1 session x 5 reps


def test_session_split_unknown_user_returns_empty_arrays(synthetic_df):
    splits = session_split(synthetic_df, "nonexistent_user", RAW_FEATURES)
    assert splits["train"].shape[0] == 0
    assert splits["val"].shape[0]   == 0
    assert splits["test"].shape[0]  == 0


def test_sample_impostors_excludes_genuine_user(synthetic_df, synthetic_user):
    impostors = sample_impostors(synthetic_df, synthetic_user, RAW_FEATURES,
                                 n=20, random_state=0)
    genuine_rows = synthetic_df[synthetic_df["subject"] == synthetic_user][RAW_FEATURES].values

    # No impostor row should exactly match any genuine row
    for imp_row in impostors:
        matches = np.all(np.isclose(genuine_rows, imp_row), axis=1)
        assert not matches.any(), "Impostor sample matches a genuine user row"


def test_sample_impostors_respects_n_upper_bound(synthetic_df, synthetic_user):
    impostors = sample_impostors(synthetic_df, synthetic_user, RAW_FEATURES,
                                 n=15, random_state=0)
    assert impostors.shape[0] <= 15
    assert impostors.shape[1] == len(RAW_FEATURES)


def test_sample_impostors_reproducible_with_seed(synthetic_df, synthetic_user):
    a = sample_impostors(synthetic_df, synthetic_user, RAW_FEATURES, n=10, random_state=7)
    b = sample_impostors(synthetic_df, synthetic_user, RAW_FEATURES, n=10, random_state=7)
    np.testing.assert_array_equal(a, b)
