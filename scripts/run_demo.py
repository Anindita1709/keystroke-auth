#!/usr/bin/env python
"""
Single-user deep dive: EDA, PCA, hyperparameter tuning, One-Class SVM vs
Binary SVM comparison — everything needed for an interview walkthrough on
ONE subject.

Usage
-----
    python scripts/run_demo.py
    python scripts/run_demo.py --user s002 --nu 0.1 --no-show
    python scripts/run_demo.py --skip-eda --skip-binary
"""
import argparse
import logging
import sys

from keystroke_auth.data import download_dataset, load_dataset
from keystroke_auth.features import engineer_features, ENGINEERED_FEATURES
from keystroke_auth.models import train_one_class_svm, tune_one_class_svm, train_binary_svm
from keystroke_auth.evaluation import compute_eer
from keystroke_auth.visualization import (
    run_eda,
    plot_user_heatmap,
    plot_pca,
    plot_tuning_heatmap,
    plot_score_distributions,
    plot_det_roc,
)

logger = logging.getLogger("keystroke_auth.demo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--user", default=None,
                        help="Subject ID to analyse (default: first user in dataset)")
    parser.add_argument("--nu", type=float, default=None,
                        help="One-Class SVM nu (default: auto-tuned via grid search)")
    parser.add_argument("--gamma", default=None,
                        help="One-Class SVM gamma (default: auto-tuned via grid search)")
    parser.add_argument("--skip-eda", action="store_true",
                        help="Skip EDA distribution / heatmap plots")
    parser.add_argument("--skip-binary", action="store_true",
                        help="Skip Binary SVM comparison (faster run)")
    parser.add_argument("--no-show", action="store_true",
                        help="Save figures without opening display windows (CI / headless)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable DEBUG-level logging")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    show = not args.no_show

    # ── 1. Data ──────────────────────────────────────────────────────────
    logger.info("[1/7] Loading dataset")
    download_dataset()
    df = load_dataset()

    # ── 2. Feature engineering ───────────────────────────────────────────
    logger.info("[2/7] Engineering features")
    df = engineer_features(df)
    features = ENGINEERED_FEATURES

    user = args.user or df["subject"].unique()[0]
    if user not in df["subject"].unique():
        logger.error("User %r not found in dataset.", user)
        return 1
    logger.info("Target user: %s", user)

    # ── 3. EDA ───────────────────────────────────────────────────────────
    if not args.skip_eda:
        logger.info("[3/7] Exploratory data analysis")
        run_eda(df, n_users=5, show=show)
        plot_user_heatmap(df, show=show)
        plot_pca(df, user, features, show=show)
    else:
        logger.info("[3/7] Skipped (--skip-eda)")

    # ── 4. Tune One-Class SVM ─────────────────────────────────────────────
    if args.nu is not None and args.gamma is not None:
        logger.info("[4/7] Using provided hyperparameters: nu=%s gamma=%s",
                    args.nu, args.gamma)
        best_params = {"nu": args.nu, "gamma": args.gamma}
    else:
        logger.info("[4/7] Tuning One-Class SVM (nu x gamma grid search)")
        best_params, tuning_df = tune_one_class_svm(df, user, features)
        plot_tuning_heatmap(tuning_df, show=show)

    # ── 5. Train + evaluate One-Class SVM ────────────────────────────────
    logger.info("[5/7] Training final One-Class SVM")
    oc_result = train_one_class_svm(df, user, features, **best_params)
    oc_metrics = compute_eer(oc_result["test_genuine"], oc_result["test_impostor"])
    logger.info("One-Class SVM — EER: %.2f%%  AUC: %.4f  Support vectors: %d",
               oc_metrics["eer"] * 100, oc_metrics["auc"], oc_result["n_sv"])
    plot_score_distributions(oc_result["test_genuine"], oc_result["test_impostor"],
                             oc_metrics["eer"], user, "One-Class SVM", show=show)

    # ── 6. Binary SVM comparison ──────────────────────────────────────────
    bin_metrics = None
    if not args.skip_binary:
        logger.info("[6/7] Training Binary SVM (GridSearchCV)")
        bin_result = train_binary_svm(df, user, features)
        bin_metrics = bin_result
        logger.info("Binary SVM — EER: %.2f%%  AUC: %.4f",
                   bin_result["eer"] * 100, bin_result["auc"])
    else:
        logger.info("[6/7] Skipped (--skip-binary)")

    # ── 7. Comparison plot ─────────────────────────────────────────────────
    logger.info("[7/7] Plotting ROC / DET comparison")
    plot_det_roc(oc_metrics, bin_metrics, user, show=show)

    logger.info("Done. Figures saved under ./figures/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
