"""
Keystroke timing feature engineering.
"""
import logging

import pandas as pd

from keystroke_auth.constants import RAW_FEATURES

logger = logging.getLogger(__name__)

# Precompute column groups once at import time
_HOLD_COLS: list[str] = [c for c in RAW_FEATURES if c.startswith("H.")]
_DD_COLS:   list[str] = [c for c in RAW_FEATURES if c.startswith("DD.")]
_UD_COLS:   list[str] = [c for c in RAW_FEATURES if c.startswith("UD.")]

#: Full feature list after engineering  (raw + derived).
ENGINEERED_FEATURES: list[str] = RAW_FEATURES + [
    "mean_hold", "std_hold",  "max_hold",
    "mean_dd",   "std_dd",    "min_dd",
    "mean_ud",   "overlap_ratio",
    "total_time", "rhythm_cv",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature          Group            What it captures
    ───────────────  ───────────────  ────────────────────────────────────────
    mean/std/max     Hold times       Key-press firmness and variability
    mean/std/min     Down-Down (DD)   Inter-key flight-time rhythm
    mean_ud          Up-Down (UD)     Release-to-press latency
    overlap_ratio    UD < 0           Rolling typist vs hunt-and-peck
    total_time       Sum of DD + H.R  Overall password entry speed
    rhythm_cv        std/mean of DD   Consistency of cadence (CV)

    """
    df = df.copy()

    # Hold time statistics
    df["mean_hold"]     = df[_HOLD_COLS].mean(axis=1)
    df["std_hold"]      = df[_HOLD_COLS].std(axis=1)
    df["max_hold"]      = df[_HOLD_COLS].max(axis=1)

    # Down-Down (flight time) statistics
    df["mean_dd"]       = df[_DD_COLS].mean(axis=1)
    df["std_dd"]        = df[_DD_COLS].std(axis=1)
    df["min_dd"]        = df[_DD_COLS].min(axis=1)

    # Up-Down statistics
    df["mean_ud"]       = df[_UD_COLS].mean(axis=1)
    # Negative UD = key released AFTER next key pressed = rolling/touch-typing
    df["overlap_ratio"] = (df[_UD_COLS] < 0).mean(axis=1)

    # Global timing
    df["total_time"]    = df[_DD_COLS].sum(axis=1) + df["H.Return"]

    # Rhythm consistency: low CV → steady typist; high CV → burst typist
    df["rhythm_cv"]     = df["std_dd"] / (df["mean_dd"].abs() + 1e-9)

    n_derived = len(ENGINEERED_FEATURES) - len(RAW_FEATURES)
    logger.info(
        "Feature engineering: %d raw + %d derived = %d total features",
        len(RAW_FEATURES), n_derived, len(ENGINEERED_FEATURES),
    )
    return df
