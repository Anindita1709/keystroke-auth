"""
Shared plotting utilities: consistent save/show behaviour and style constants.

Keeping this separate avoids repeating the same savefig/show boilerplate in
every plotting function across the package.
"""
import logging
from pathlib import Path

import matplotlib.pyplot as plt

from keystroke_auth.constants import BLUE, ORANGE, RED, GREEN  # noqa: F401  (re-exported)

logger = logging.getLogger(__name__)

#: Default directory where all figures are written.
FIGURE_DIR = Path("figures")


def save_and_show(fig: plt.Figure, filename: str, show: bool = True) -> Path:
    """
    Save a figure to FIGURE_DIR and optionally display it.

    Centralising this means changing DPI, output format, or disabling
    plt.show() (e.g. in CI / headless tests) only requires editing one place.

    Parameters
    ----------
    fig      : Matplotlib Figure to save.
    filename : Output filename (e.g. "eda_distributions.png").
    show     : If False, skip plt.show() — useful for headless/CI runs.

    Returns
    -------
    Path to the saved file.
    """
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURE_DIR / filename

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info("Saved figure → %s", out_path)

    if show:
        plt.show()
    plt.close(fig)

    return out_path
