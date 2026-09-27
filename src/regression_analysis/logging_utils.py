"""Thin wrapper around :mod:`logging` so modules share one configured logger."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging once, writing plain messages to stdout.

    The stream is captured at configuration time so that later redirection of
    ``sys.stdout`` (used to tee the log to a file) does not duplicate output.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(level=level, format="%(message)s", stream=sys.__stdout__)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, configuring logging on first use."""
    configure_logging()
    return logging.getLogger(name)

