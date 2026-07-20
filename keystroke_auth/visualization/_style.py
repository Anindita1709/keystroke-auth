
import logging
from pathlib import Path

import matplotlib.pyplot as plt

from keystroke_auth.constants import BLUE, ORANGE, RED, GREEN  # noqa: F401  (re-exported)

logger = logging.getLogger(__name__)

# Default directory where all figures are written.
FIGURE_DIR = Path("figures")

def save_and_show(fig: plt.Figure, filename: str, show: bool = True) -> Path:
    
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURE_DIR / filename

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info("Saved figure → %s", out_path)

    if show:
        plt.show()
    plt.close(fig)

    return out_path
