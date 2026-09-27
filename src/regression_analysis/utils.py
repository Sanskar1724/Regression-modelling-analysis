"""Shared helpers: metric computation, figure I/O and console formatting."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import FIG_DIR
from .logging_utils import get_logger

LOGGER = get_logger(__name__)

#: Column order used by every comparison table in this project.
METRIC_COLUMNS = ["R2", "Adj_R2", "MAE", "MSE", "RMSE", "n_features"]


def section(title: str, width: int = 78) -> None:
    """Print a full-width section banner."""
    LOGGER.info("")
    LOGGER.info("=" * width)
    LOGGER.info(title)
    LOGGER.info("=" * width)


def subsection(title: str) -> None:
    """Print a sub-section heading."""
    LOGGER.info("")
    LOGGER.info("--- %s ---", title)


def save_figure(name: str) -> None:
    """Persist the current matplotlib figure and release the canvas."""
    import matplotlib.pyplot as plt

    plt.savefig(FIG_DIR / name)
    plt.close()
    LOGGER.info("[figure saved] outputs/figures/%s", name)


def compute_metrics(y_true, y_pred, n_features: int) -> dict:
    """Return R2, Adjusted R2, MAE, MSE and RMSE for a prediction.

    Adjusted R2 penalises the parameter count so that models can be compared
    fairly regardless of how many terms they use.
    """
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "R2": r2,
        "Adj_R2": adjusted_r2,
        "MAE": mae,
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "n_features": n_features,
    }
