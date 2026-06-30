"""Tests for keystroke_auth.evaluation.eer."""
import numpy as np

from keystroke_auth.evaluation.eer import compute_eer


def test_eer_perfect_separation_is_near_zero():
    # Genuine scores all high, impostor scores all low -> no overlap -> EER ~ 0
    rng = np.random.default_rng(0)
    genuine  = rng.normal(loc=5.0, scale=0.1, size=200)
    impostor = rng.normal(loc=-5.0, scale=0.1, size=200)

    result = compute_eer(genuine, impostor)

    assert result["eer"] < 0.02
    assert result["auc"] > 0.99


def test_eer_identical_distributions_is_near_half():
    # Genuine and impostor scores drawn from the SAME distribution ->
    # classifier has no discriminative power -> EER ~ 0.5, AUC ~ 0.5
    rng = np.random.default_rng(1)
    genuine  = rng.normal(loc=0.0, scale=1.0, size=500)
    impostor = rng.normal(loc=0.0, scale=1.0, size=500)

    result = compute_eer(genuine, impostor)

    assert 0.35 < result["eer"] < 0.65
    assert 0.40 < result["auc"] < 0.60


def test_eer_returns_all_expected_keys():
    genuine  = np.array([1.0, 2.0, 3.0])
    impostor = np.array([-1.0, -2.0, -3.0])
    result = compute_eer(genuine, impostor)

    expected_keys = {"eer", "eer_threshold", "auc", "fpr", "tpr", "frr", "thresholds"}
    assert expected_keys.issubset(result.keys())


def test_eer_is_between_zero_and_one_for_realistic_overlap():
    rng = np.random.default_rng(2)
    genuine  = rng.normal(loc=1.0, scale=1.0, size=300)
    impostor = rng.normal(loc=-1.0, scale=1.0, size=300)

    result = compute_eer(genuine, impostor)

    assert not np.isnan(result["eer"])
    assert 0.0 <= result["eer"] <= 1.0


def test_eer_higher_auc_implies_lower_eer():
    rng = np.random.default_rng(3)

    # Well-separated case
    gen_easy = rng.normal(3.0, 0.5, 200)
    imp_easy = rng.normal(-3.0, 0.5, 200)
    easy = compute_eer(gen_easy, imp_easy)

    # Heavily overlapping case
    gen_hard = rng.normal(0.2, 1.0, 200)
    imp_hard = rng.normal(-0.2, 1.0, 200)
    hard = compute_eer(gen_hard, imp_hard)

    assert easy["auc"] > hard["auc"]
    assert easy["eer"] < hard["eer"]
