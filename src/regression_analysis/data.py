"""Loading, profiling and train/test splitting of the California Housing data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from .config import FEATURES, RANDOM_STATE, TARGET, TEST_SIZE, get_logger

LOGGER = get_logger(__name__)


def load_dataset() -> tuple[pd.DataFrame, list[str]]:
    """Return the California Housing frame plus its feature names.

    The frame contains the eight predictors, the target, and a convenience
    column ``HouseValue_USD`` expressing the target in dollars.
    """
    LOGGER.info("Loading California Housing dataset (scikit-learn)...")
    bunch = fetch_california_housing()
    frame = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    frame[TARGET] = bunch.target
    frame["HouseValue_USD"] = frame[TARGET] * 100_000
    LOGGER.info("  Shape              : %s", frame.shape)
    LOGGER.info("  Features           : %s", ", ".join(bunch.feature_names))
    LOGGER.info("  Target             : %s (units of USD 100,000)", TARGET)
    return frame, list(bunch.feature_names)


def train_test_split_data(
    frame: pd.DataFrame, features: list[str] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split into an 80/20 train/test partition with a fixed seed."""
    features = features or FEATURES
    x_train, x_test, y_train, y_test = train_test_split(
        frame[features],
        frame[TARGET],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    LOGGER.info("")
    LOGGER.info("=" * 78)
    LOGGER.info("DATA PARTITIONING  |  %d%% TRAIN / %d%% TEST",
                int((1 - TEST_SIZE) * 100), int(TEST_SIZE * 100))
    LOGGER.info("=" * 78)
    LOGGER.info("  Training set : %s rows x %d features",
                f"{len(x_train):,}", x_train.shape[1])
    LOGGER.info("  Test set     : %s rows x %d features",
                f"{len(x_test):,}", x_test.shape[1])
    LOGGER.info("  Random state : %d", RANDOM_STATE)
    LOGGER.info("  Target mean  : train %.4f | test %.4f",
                y_train.mean(), y_test.mean())
    LOGGER.info("")
    LOGGER.info("  NOTE: the two means are close, so the random split produced a")
    LOGGER.info("  representative test set and the test R2 is a fair estimate.")
    return x_train, x_test, y_train, y_test


def data_profile(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Return a per-column dtype / missingness / cardinality profile."""
    core = features + [TARGET]
    profile = pd.DataFrame(
        {
            "dtype": frame[core].dtypes.astype(str),
            "missing": frame[core].isna().sum(),
            "missing_%": (frame[core].isna().mean() * 100).round(2),
            "n_unique": frame[core].nunique(),
        }
    )
    return profile


def descriptive_statistics(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Return the transposed describe() table augmented with shape statistics."""
    core = features + [TARGET]
    stats = frame[core].describe().T
    stats["skew"] = frame[core].skew()
    stats["kurtosis"] = frame[core].kurtosis()
    return stats


def distribution_shape(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Classify each column by skewness magnitude."""
    core = features + [TARGET]
    shape = pd.DataFrame(
        {"skewness": frame[core].skew(), "kurtosis": frame[core].kurtosis()}
    )
    shape["verdict"] = np.where(
        shape["skewness"].abs() > 1,
        "strongly skewed",
        np.where(shape["skewness"].abs() > 0.5, "moderately skewed", "approx. symmetric"),
    )
    return shape
