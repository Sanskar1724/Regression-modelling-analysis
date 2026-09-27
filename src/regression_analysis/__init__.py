"""Regression analysis on the California Housing dataset.

A reproducible implementation of the ten mandatory regression lab tasks:
exploratory analysis, correlation, outlier detection, VIF, nine regression
models, model comparison, assumption testing, coefficient interpretation,
cross-validated tuning and justified final model selection.
"""

from __future__ import annotations

__version__ = "1.0.0"
__author__ = "Sanskar1724"

from .config import FEATURES, RANDOM_STATE, TARGET
from .pipeline import run

__all__ = ["FEATURES", "RANDOM_STATE", "TARGET", "run", "__version__"]
