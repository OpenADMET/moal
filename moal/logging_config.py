"""Centralized logging suppression for third-party libraries.

Call ``suppress_noisy_loggers()`` once at the start of a campaign to prevent
PyTorch Lightning, RDKit, matplotlib, and other libraries from cluttering
the console output with verbose progress lines, deprecation notices, and
internal status messages.
"""

from __future__ import annotations

import logging


def suppress_noisy_loggers() -> None:
    """Set noisy third-party loggers to WARNING level and silence RDKit's C++ logger.

    Safe to call multiple times (idempotent).
    """
    _silence_rdkit()
    _set_level(logging.WARNING, [
        "lightning",
        "lightning.pytorch",
        "lightning.pytorch.utilities",
        "lightning.pytorch.trainer",
        "lightning.pytorch.accelerators",
        "lightning.pytorch.callbacks",
        "lightning.pytorch.core",
        "torch",
        "matplotlib",
        "matplotlib.font_manager",
        "PIL",
        "urllib3",
        "filelock",
        "fsspec",
        "importlib_metadata",
        "pkg_resources",
    ])


def _silence_rdkit() -> None:
    """Disable RDKit's C++ logger and set the Python rdkit logger to WARNING."""
    try:
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        pass
    # Also silence the Python-side rdkit logger.
    logging.getLogger("rdkit").setLevel(logging.WARNING)


def _set_level(level: int, names: list[str]) -> None:
    """Set the logging level for a list of logger names.

    Parameters
    ----------
    level : int
        Logging level constant (e.g., ``logging.WARNING``).
    names : list[str]
        Logger names to configure.
    """
    for name in names:
        logging.getLogger(name).setLevel(level)
