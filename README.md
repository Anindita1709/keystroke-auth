# Keystroke Dynamics Authentication with SVM

Authenticate users from *how they type* a fixed password, not just whether
they know it. Built on the CMU Keystroke Dynamics benchmark dataset, this
project compares **One-Class SVM** (novelty detection, trained only on the
genuine user) against **Binary SVM** (genuine vs. impostor classification)
as biometric authenticators.

[![CI](https://github.com/yourusername/keystroke-auth/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/keystroke-auth/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Why this problem is interesting

Most ML authentication demos rely on something a user *has* (a token) or
*knows* (a password). Keystroke dynamics authenticates based on something a
user *is* — their physical typing rhythm — using only software, no extra
hardware. It sits in the same family as gait recognition and signature
dynamics, and is used in real fraud-detection and insider-threat systems.

The dataset: 51 subjects each typed the password `.tie5Roanl` 400 times
(8 sessions x 50 repetitions), with three timing measurements captured per
keystroke transition — hold time, key-to-key down-down latency, and
key-to-key up-down latency.

> Killourhy, K.S. and Maxion, R.A., "Comparing Anomaly Detectors for
> Keystroke Dynamics," *IEEE/IFIP Dependable Systems and Networks*, 2009.

## Architecture

```
keystroke_auth/
├── constants.py          # dataset URL, raw feature names, split boundaries
├── data/
│   ├── loader.py          # download_dataset(), load_dataset()
│   └── splits.py          # session_split(), sample_impostors()
├── features/
│   └── engineering.py     # engineer_features() — 31 raw -> 41 features
├── models/
│   ├── one_class.py        # train_one_class_svm(), tune_one_class_svm()
│   └── binary.py            # train_binary_svm() (GridSearchCV baseline)
├── evaluation/
│   ├── eer.py                # compute_eer() — FAR/FRR crossing via brentq
│   └── system.py              # evaluate_all_users() — system-wide EER
└── visualization/
    ├── eda_plots.py            # timing distributions, user heatmap, PCA
    ├── model_plots.py           # tuning heatmap, score dist, ROC/DET
    └── system_plots.py          # EER distribution across all subjects

scripts/
├── run_demo.py            # single-user deep dive (CLI)
└── run_system_eval.py     # full 51-user system evaluation (CLI)

tests/                     # 26 tests, synthetic data, no network required
```

Each module has a single responsibility and no function depends on global
state — every model/eval function takes the dataframe and feature list as
explicit arguments, which is what makes the test suite able to run fully
offline against synthetic fixtures.

## Installation

```bash
git clone https://github.com/yourusername/keystroke-auth.git
cd keystroke-auth
pip install -e ".[dev]"
```

Requires Python 3.10+. Core dependencies: numpy, pandas, scikit-learn,
matplotlib, scipy.

## Usage

### Single-user deep dive

Runs EDA, PCA visualization, nu/gamma hyperparameter tuning, trains both
models, and plots ROC/DET comparison curves for one subject:

```bash
python scripts/run_demo.py
python scripts/run_demo.py --user s002 --nu 0.1 --gamma scale
python scripts/run_demo.py --skip-eda --skip-binary   # fast path
```

The dataset auto-downloads to the working directory on first run.

### Full system evaluation

Runs One-Class SVM authentication for all 51 subjects and reports
aggregate EER statistics:

```bash
python scripts/run_system_eval.py
python scripts/run_system_eval.py --nu 0.1 --gamma scale --output results.csv
```

Both scripts accept `--no-show` (save figures without opening display
windows — useful over SSH or in CI) and `-v` for debug logging.

### As a library

```python
from keystroke_auth.data import download_dataset, load_dataset
from keystroke_auth.features import engineer_features, ENGINEERED_FEATURES
from keystroke_auth.models import train_one_class_svm
from keystroke_auth.evaluation import compute_eer

download_dataset()
df = engineer_features(load_dataset())

result = train_one_class_svm(df, user="s002", features=ENGINEERED_FEATURES, nu=0.1)
metrics = compute_eer(result["test_genuine"], result["test_impostor"])
print(f"EER: {metrics['eer']*100:.2f}%  AUC: {metrics['auc']:.4f}")
```

## Methodology

**Session-based splitting, not random.** Sessions 1-5 train, 6-7 validate,
8 tests. Keystroke timing improves slightly as users get comfortable in
later sessions; a random split would leak that future consistency into
training and inflate every reported number. This protocol matches the
original CMU benchmark paper.

**Equal Error Rate (EER)** is the headline metric, not accuracy — the
operating point where False Acceptance Rate equals False Rejection Rate.
`evaluation/eer.py` finds this via `scipy.optimize.brentq` root-finding on
the interpolated FAR/FRR curves rather than just picking the closest
threshold index, which matters when the curves are sparse.

**One-Class SVM vs. Binary SVM** is the central comparison:

| | One-Class SVM | Binary SVM |
|---|---|---|
| Trains on | Genuine user only | Genuine + sampled impostors |
| Enrollment realism | High — no impostor data needed | Low — assumes impostor data exists upfront |
| Typical EER | Higher | Lower |
| Failure mode | N/A | Open-set problem: vulnerable to impostors unlike any seen in training |

Reported EER in the literature for classical detectors on this dataset
commonly falls in the 10-20% range — useful as a sanity check for whether
your own numbers are in a reasonable ballpark, though figures vary by
detector and feature set.

## Testing

```bash
pytest tests/ -v
ruff check .
```

The suite (26 tests) uses a synthetic fixture matching the CMU CSV schema
(see `tests/conftest.py`) so it runs in about a second with no network
access. It covers feature engineering invariants, split correctness
(disjoint partitions, impostor exclusion, reproducibility), EER behavior
at the extremes (perfect separation -> EER tends to 0, identical
distributions -> EER tends to 0.5), and end-to-end smoke tests for both
models.

## Extending this project

A few directions that make good follow-up work: free-text typing instead
of a fixed password (requires variable-length sequence features rather
than fixed 31-dim vectors), continuous re-authentication during a session
rather than one-shot login, or comparing against an Isolation Forest or
Gaussian Mixture Model baseline alongside the two SVM variants here.

## License

MIT — see [LICENSE](LICENSE).
