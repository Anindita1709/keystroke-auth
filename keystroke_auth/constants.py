"""
Global constants shared across all modules.
"""
from pathlib import Path

# ── Dataset ────────────────────────────────────────────────────────────────
DATASET_URL  = "http://www.cs.cmu.edu/~keystroke/DSL-StrongPasswordData.csv"
DATASET_FILE = Path("DSL-StrongPasswordData.csv")

# ── Raw timing features (31 columns) for password ".tie5Roanl" ─────────────
# H.*     = Hold time    (key pressed  → key released)
# DD.*.*  = Down-Down   (key A pressed → key B pressed)
# UD.*.*  = Up-Down     (key A released → key B pressed)  can be negative
RAW_FEATURES: list[str] = [
    "H.period",
    "DD.period.t",     "UD.period.t",
    "H.t",
    "DD.t.i",          "UD.t.i",
    "H.i",
    "DD.i.e",          "UD.i.e",
    "H.e",
    "DD.e.five",       "UD.e.five",
    "H.five",
    "DD.five.Shift.r", "UD.five.Shift.r",
    "H.Shift.r",
    "DD.Shift.r.o",    "UD.Shift.r.o",
    "H.o",
    "DD.o.a",          "UD.o.a",
    "H.a",
    "DD.a.n",          "UD.a.n",
    "H.n",
    "DD.n.l",          "UD.n.l",
    "H.l",
    "DD.l.Return",     "UD.l.Return",
    "H.Return",
]

# ── Session split boundaries (dataset has 8 sessions per user) ─────────────
TRAIN_SESSION_END: int = 5   # sessions 1–5  → training
VAL_SESSION_END:   int = 7   # sessions 6–7  → validation / tuning
                             # session  8    → test (never touched until final eval)

# ── Plot colours ───────────────────────────────────────────────────────────
BLUE   = "#4472C4"
ORANGE = "#ED7D31"
RED    = "#C00000"
GREEN  = "#70AD47"
