#!/usr/bin/env python
"""
System-wide evaluation: run One-Class SVM authentication for every subject
in the CMU dataset and report aggregate EER statistics.

Usage
-----
    python scripts/run_system_eval.py
    python scripts/run_system_eval.py --nu 0.1 --gamma scale --no-show
    python scripts/run_system_eval.py --output results/full_run.csv
"""
import argparse
import logging
import sys

from keystroke_auth.data import download_dataset, load_dataset
from keystroke_auth.features import engineer_features, ENGINEERED_FEATURES
from keystroke_auth.evaluation import evaluate_all_users
from keystroke_auth.visualization import plot_system_eer

logger = logging.getLogger("keystroke_auth.system_eval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nu", type=float, default=0.1,
                        help="One-Class SVM nu, applied uniformly across all "
                             "users (default: 0.1). Per-user tuning would "
                             "inflate system-wide results unrealistically.")
    parser.add_argument("--gamma", default="scale",
                        help="One-Class SVM gamma (default: 'scale')")
    parser.add_argument("--output", default="system_results.csv",
                        help="Path to write per-user EER results CSV")
    parser.add_argument("--no-show", action="store_true",
                        help="Save figures without opening display windows")
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

    logger.info("[1/3] Loading dataset")
    download_dataset()
    df = load_dataset()

    logger.info("[2/3] Engineering features")
    df = engineer_features(df)

    logger.info("[3/3] Evaluating all %d users (this may take a few minutes)",
               df["subject"].nunique())
    # gamma may arrive as a numeric string (e.g. "0.05") — cast if possible
    gamma = args.gamma
    try:
        gamma = float(gamma)
    except ValueError:
        pass  # keep as string, e.g. "scale" / "auto"

    results_df = evaluate_all_users(
        df, ENGINEERED_FEATURES,
        nu=args.nu, gamma=gamma,
        save_path=args.output,
    )

    plot_system_eer(results_df, show=not args.no_show)

    logger.info("Done. Results written to %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
