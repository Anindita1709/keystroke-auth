
from keystroke_auth.evaluation.eer import compute_eer

__all__ = ["compute_eer", "evaluate_all_users"]


def __getattr__(name):
    if name == "evaluate_all_users":
        from keystroke_auth.evaluation.system import evaluate_all_users
        return evaluate_all_users
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
