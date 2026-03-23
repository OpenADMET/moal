"""Centralized logging suppression for third-party libraries.

Call ``suppress_noisy_loggers()`` once at the start of a campaign to prevent
PyTorch Lightning, RDKit, matplotlib, and other libraries from cluttering
the console output with verbose progress lines, deprecation notices, and
internal status messages.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterator
from contextlib import contextmanager


def suppress_noisy_loggers() -> None:
    """Set noisy third-party loggers to WARNING level and silence RDKit's C++ logger.

    Safe to call multiple times (idempotent).
    """
    _silence_rdkit()
    _silence_third_party_warnings()
    _set_level(
        logging.WARNING,
        [
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
            "kaleido",
            "choreographer",
            "choreographer.browsers.chromium",
            "choreographer.browser_async",
            "choreographer.utils._tmpfile",
        ],
    )


def _silence_rdkit() -> None:
    """Disable RDKit's C++ logger and set the Python rdkit logger to WARNING."""
    try:
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        pass
    # Also silence the Python-side rdkit logger.
    logging.getLogger("rdkit").setLevel(logging.WARNING)


def _silence_third_party_warnings() -> None:
    """Suppress FutureWarnings from third-party libraries that use deprecated torch internals.

    lightning 2.6.1 uses ``LeafSpec()`` from ``torch.utils._pytree``, which torch 2.10+
    deprecated. The warning surfaces from lightning's own ``_pytree.py`` (the call site).
    This is an upstream Lightning bug; remove this filter once Lightning ships a fix.
    """
    warnings.filterwarnings(
        "ignore",
        message=r"`isinstance\(treespec, LeafSpec\)` is deprecated",
        category=FutureWarning,
        module=r"lightning\.pytorch\.utilities\._pytree",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"The .* does not have many workers",
        category=UserWarning,
        module=r"lightning\.pytorch",
    )


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


@contextmanager
def temporary_log_level(level: int, names: list[str]) -> Iterator[None]:
    """Temporarily raise logger thresholds while a progress display is active.

    Keeps informational logs from redrawing into Rich live output while still
    allowing warnings and errors to surface. Restores original levels on exit,
    even if an exception is raised.

    Parameters
    ----------
    level : int
        Logging level to set temporarily (e.g., ``logging.WARNING``).
    names : list[str]
        Logger names whose levels will be raised for the duration of the block.
    """
    loggers = [logging.getLogger(name) for name in names]
    previous_levels = [logger.level for logger in loggers]
    try:
        for logger in loggers:
            logger.setLevel(level)
        yield
    finally:
        for logger, previous_level in zip(loggers, previous_levels, strict=True):
            logger.setLevel(previous_level)
