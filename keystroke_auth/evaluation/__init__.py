"""
evaluation.eer is imported eagerly since it has no internal dependencies.

evaluation.system is imported LAZILY (via module __getattr__, PEP 562)
because it depends on keystroke_auth.models.one_class, which itself depends
on evaluation.eer. Eagerly importing system here would force Python to
import models.one_class WHILE this package's __init__.py is still running,
before eer's compute_eer has finished binding — a circular import.
Deferring the import until evaluate_all_users is actually accessed breaks
the cycle without changing any public-facing API.
"""
from keystroke_auth.evaluation.eer import compute_eer

__all__ = ["compute_eer", "evaluate_all_users"]


def __getattr__(name):
    if name == "evaluate_all_users":
        from keystroke_auth.evaluation.system import evaluate_all_users
        return evaluate_all_users
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
