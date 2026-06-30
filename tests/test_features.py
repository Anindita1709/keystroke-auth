"""Tests for keystroke_auth.features.engineering."""

from keystroke_auth.constants import RAW_FEATURES
from keystroke_auth.features import engineer_features, ENGINEERED_FEATURES


def test_engineer_features_adds_expected_columns(synthetic_df):
    out = engineer_features(synthetic_df)

    expected_new_cols = {
        "mean_hold", "std_hold", "max_hold",
        "mean_dd", "std_dd", "min_dd",
        "mean_ud", "overlap_ratio",
        "total_time", "rhythm_cv",
    }
    assert expected_new_cols.issubset(set(out.columns))


def test_engineered_features_list_matches_dataframe_columns(synthetic_df):
    out = engineer_features(synthetic_df)
    # Every name in the canonical feature list must actually exist as a column
    missing = [f for f in ENGINEERED_FEATURES if f not in out.columns]
    assert not missing, f"Missing engineered columns: {missing}"


def test_engineer_features_does_not_mutate_input(synthetic_df):
    original_cols = set(synthetic_df.columns)
    _ = engineer_features(synthetic_df)
    assert set(synthetic_df.columns) == original_cols, (
        "engineer_features must not mutate the input DataFrame in place"
    )


def test_overlap_ratio_in_valid_range(synthetic_df):
    out = engineer_features(synthetic_df)
    assert out["overlap_ratio"].between(0, 1).all()


def test_total_time_is_positive(synthetic_df):
    out = engineer_features(synthetic_df)
    # total_time sums non-negative DD columns + non-negative H.Return
    assert (out["total_time"] >= 0).all()


def test_raw_features_count():
    # Sanity check: exactly 31 raw timing columns for the .tie5Roanl password
    assert len(RAW_FEATURES) == 31
