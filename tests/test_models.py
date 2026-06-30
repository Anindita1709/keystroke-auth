"""
Smoke tests for One-Class SVM and Binary SVM training.

These are not meant to assert specific EER values (the synthetic fixture
is tiny and random), but to guarantee the full train/evaluate code path
runs end-to-end without raising, returns correctly-shaped outputs, and
that the StandardScaler is fit only on training data.
"""
import numpy as np

from keystroke_auth.constants import RAW_FEATURES
from keystroke_auth.models.one_class import train_one_class_svm, tune_one_class_svm
from keystroke_auth.models.binary import train_binary_svm


def test_train_one_class_svm_returns_expected_keys(synthetic_df, synthetic_user):
    result = train_one_class_svm(synthetic_df, synthetic_user, RAW_FEATURES,
                                 nu=0.2, gamma="scale")

    expected_keys = {"model", "scaler", "n_sv",
                     "val_genuine", "val_impostor",
                     "test_genuine", "test_impostor"}
    assert expected_keys.issubset(result.keys())
    assert result["n_sv"] > 0
    assert len(result["test_genuine"]) == 5    # 1 test session x 5 reps
    assert len(result["test_impostor"]) == 100 # n=100 requested in train_one_class_svm


def test_one_class_svm_scaler_fit_only_on_train(synthetic_df, synthetic_user):
    result = train_one_class_svm(synthetic_df, synthetic_user, RAW_FEATURES, nu=0.2)
    scaler = result["scaler"]

    from keystroke_auth.data.splits import session_split
    train_data = session_split(synthetic_df, synthetic_user, RAW_FEATURES)["train"]

    # The scaler's learned mean should equal the training data's mean,
    # NOT a mean computed over train+val+test combined.
    np.testing.assert_allclose(scaler.mean_, train_data.mean(axis=0), rtol=1e-6)


def test_tune_one_class_svm_returns_valid_params(synthetic_df, synthetic_user):
    best_params, results_df = tune_one_class_svm(
        synthetic_df, synthetic_user, RAW_FEATURES,
        nu_grid=[0.05, 0.1, 0.2],
        gamma_grid=["scale", 0.01],
    )

    assert "nu" in best_params and "gamma" in best_params
    assert best_params["nu"] in [0.05, 0.1, 0.2]
    assert len(results_df) > 0
    assert {"nu", "gamma", "eer", "auc"}.issubset(results_df.columns)


def test_train_binary_svm_returns_expected_keys(synthetic_df, synthetic_user):
    # Small param grid to keep the smoke test fast
    small_grid = {"C": [1, 10], "gamma": ["scale"], "kernel": ["rbf"]}
    result = train_binary_svm(synthetic_df, synthetic_user, RAW_FEATURES,
                              param_grid=small_grid, cv_folds=2)

    expected_keys = {"model", "scaler", "best_params", "cv_auc",
                     "eer", "auc", "fpr", "tpr", "frr"}
    assert expected_keys.issubset(result.keys())
    assert 0.0 <= result["auc"] <= 1.0


def test_binary_svm_predict_proba_available(synthetic_df, synthetic_user):
    small_grid = {"C": [1], "gamma": ["scale"], "kernel": ["rbf"]}
    result = train_binary_svm(synthetic_df, synthetic_user, RAW_FEATURES,
                              param_grid=small_grid, cv_folds=2)
    # probability=True must have been set for predict_proba to exist
    assert hasattr(result["model"], "predict_proba")
