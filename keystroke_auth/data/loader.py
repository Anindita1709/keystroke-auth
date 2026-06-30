"""Dataset download and loading utilities."""
import logging
import urllib.request
from pathlib import Path

import pandas as pd

from keystroke_auth.constants import DATASET_URL, DATASET_FILE, RAW_FEATURES

logger = logging.getLogger(__name__)


def download_dataset(
    url: str = DATASET_URL,
    filepath: Path | str = DATASET_FILE,
) -> Path:
    """
    Download the CMU Keystroke Dynamics dataset if not already present.

    Parameters
    ----------
    url      : Source URL of the CSV file.
    filepath : Local destination path.

    Returns
    -------
    Path to the local file.

    Raises
    ------
    RuntimeError if download fails.
    """
    filepath = Path(filepath)
    if filepath.exists():
        logger.info("Dataset already present at %s", filepath)
        return filepath

    logger.info("Downloading dataset from %s …", url)
    try:
        urllib.request.urlretrieve(str(url), filepath)
        logger.info("Download complete → %s", filepath)
    except Exception as exc:
        raise RuntimeError(
            f"Download failed: {exc}\n"
            f"Download manually from: {url}"
        ) from exc

    return filepath


def load_dataset(filepath: Path | str = DATASET_FILE) -> pd.DataFrame:
    """
    Load the CMU dataset CSV into a DataFrame and log a summary.

    Expected columns: subject, sessionIndex, rep, <31 timing features>

    Parameters
    ----------
    filepath : Path to the local CSV file.

    Returns
    -------
    pd.DataFrame with all rows and raw timing columns.

    Raises
    ------
    FileNotFoundError if the CSV is missing (call download_dataset first).
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found at {filepath}. "
            "Run download_dataset() first."
        )

    df = pd.read_csv(filepath)

    logger.info(
        "Loaded %s — subjects: %d, rows: %d, timing features: %d",
        filepath.name,
        df["subject"].nunique(),
        len(df),
        len(RAW_FEATURES),
    )
    return df
