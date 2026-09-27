"""Project-wide configuration and path resolution.

All paths are derived from this file's location so the project is portable:
cloning the repository anywhere and running the entry point works unchanged.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
# src/regression_analysis/config.py -> parents[2] is the repository root.
PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUTPUTS_DIR / "figures"
TAB_DIR = OUTPUTS_DIR / "tables"
DOCS_DIR = PROJECT_ROOT / "docs"

for _directory in (FIG_DIR, TAB_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# ANALYSIS SETTINGS
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

TARGET = "MedHouseVal"
"""Target column: median house value, expressed in units of USD 100,000."""

FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]
"""The eight predictors supplied by ``fetch_california_housing``."""

#: Hard ceiling present in the source data (USD 500,005). Predictions for more
#: expensive blocks are structurally biased downwards because of it.
TARGET_CEILING = 5.0005

# Regularisation grids searched during cross-validation.
# Ridge spans eight orders of magnitude in 0.25 steps (41 candidates), which is
# the standard sweep and matters here: a coarse integer-exponent grid previously
# offered only 9 candidates and missed the optimum.
RIDGE_ALPHAS = tuple(10.0 ** (step / 4.0) for step in range(-16, 17))
LASSO_ALPHAS = tuple(10.0 ** (step / 2.0) for step in range(-8, 5))
ELASTICNET_ALPHAS = tuple(10.0 ** (step / 2.0) for step in range(-8, 5))
ELASTICNET_L1_RATIOS = tuple(round(0.05 * i, 2) for i in range(1, 20))


def configure_logging(level: int = logging.INFO) -> None:
    """Send INFO messages to stdout so the run log can be captured directly."""
    logging.basicConfig(
        level=level,
        format="%(message)s",
        stream=os.sys.stdout,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger."""
    return logging.getLogger(name)
