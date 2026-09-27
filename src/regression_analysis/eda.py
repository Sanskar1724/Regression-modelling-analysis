"""Task 1 - Exploratory Data Analysis.

Profiles the dataset, characterises each marginal distribution and produces
the histogram, boxplot, ceiling, geospatial and missing-value figures.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import TAB_DIR, TARGET
from .logging_utils import get_logger
from .utils import save_figure, subsection

LOGGER = get_logger(__name__)


def run_eda(frame: pd.DataFrame, features: list[str]) -> dict:
    """Execute the EDA stage and return the intermediate tables."""
    results: dict[str, pd.DataFrame] = {}
    core = features + [TARGET]

    subsection("1.1 Dataset shape, features and target")
    LOGGER.info("  Rows x columns (incl. target) : %s", frame.shape)
    LOGGER.info("  Memory usage                  : %.1f KB",
                frame.memory_usage(deep=True).sum() / 1024)
    LOGGER.info("  Target ceiling                : %.4f (USD 500,005)",
                frame[TARGET].max())

    subsection("1.2 Data types, null counts and uniqueness")
    profile = pd.DataFrame({
        "dtype": frame[core].dtypes.astype(str),
        "missing": frame[core].isna().sum(),
        "missing_%": (frame[core].isna().mean() * 100).round(2),
        "n_unique": frame[core].nunique(),
    })
    LOGGER.info("%s", profile.to_string())
    LOGGER.info("")
    LOGGER.info("  Total missing cells : %d", int(frame.isna().sum().sum()))
    LOGGER.info("  Infinite values     : %d",
                int(np.isinf(frame[core].to_numpy()).sum()))
    LOGGER.info("")
    LOGGER.info("  COMMENT: no missing or infinite values, so no imputation or")
    LOGGER.info("  row-dropping is required. The dataset is completely rectangular.")
    profile.to_csv(TAB_DIR / "01_dataset_profile.csv")
    results["profile"] = profile

    subsection("1.3 Descriptive statistics")
    stats = frame[core].describe().T
    stats["skew"] = frame[core].skew()
    stats["kurtosis"] = frame[core].kurtosis()
    LOGGER.info("%s", stats[["mean", "std", "min", "25%", "50%", "75%", "max",
                             "skew", "kurtosis"]].round(3).to_string())
    LOGGER.info("")
    LOGGER.info("  COMMENT: the target is capped at %.4f - a hard ceiling in the source",
                frame[TARGET].max())
    LOGGER.info("  data (top-coded at USD 500,005). This censored region is the single")
    LOGGER.info("  most important fact for model choice: no model can predict above it,")
    LOGGER.info("  so predictions for expensive blocks are biased downwards by design.")
    stats.to_csv(TAB_DIR / "02_descriptive_statistics.csv")
    results["descriptive"] = stats

    subsection("1.4 Distribution shape (skewness and kurtosis)")
    shape = pd.DataFrame({
        "skewness": frame[core].skew(),
        "kurtosis": frame[core].kurtosis(),
    })
    shape["verdict"] = np.where(
        shape["skewness"].abs() > 1, "strongly skewed",
        np.where(shape["skewness"].abs() > 0.5, "moderately skewed", "approx. symmetric"),
    )
    LOGGER.info("%s", shape.round(3).to_string())
    LOGGER.info("")
    LOGGER.info("  COMMENT: AveRooms, AveBedrms, Population and above all AveOccup are")
    LOGGER.info("  strongly right-skewed because they are per-block RATIOS - one large")
    LOGGER.info("  household in a small block produces an extreme value. This justifies")
    LOGGER.info("  the log-transform option in the brief and explains the residual Q-Q")
    LOGGER.info("  departure found in the assumption tests.")
    shape.to_csv(TAB_DIR / "03_distribution_shape.csv")
    results["shape"] = shape

    _plot_histograms(frame, core)
    _plot_boxplots(frame, core)
    _plot_target_ceiling(frame)
    _plot_geospatial(frame)
    _plot_missing(frame, core)
    _target_bands(frame, results)

    return results


def _plot_histograms(frame: pd.DataFrame, core: list[str]) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    for ax, column in zip(axes.flat, core):
        ax.hist(frame[column], bins=40, color="#4C72B0", edgecolor="white", linewidth=0.4)
        ax.set_title(f"{column}\nskew={frame[column].skew():.2f}", fontsize=10)
        ax.set_ylabel("Frequency")
    fig.suptitle("Figure 1  -  Histograms of all predictors and the target",
                 fontsize=14, fontweight="bold")
    save_figure("fig01_histograms.png")


def _plot_boxplots(frame: pd.DataFrame, core: list[str]) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    for ax, column in zip(axes.flat, core):
        ax.boxplot(frame[column], vert=True, patch_artist=True,
                   boxprops=dict(facecolor="#55A868", alpha=0.7),
                   medianprops=dict(color="red", linewidth=2),
                   flierprops=dict(marker="o", markerfacecolor="red",
                                   markersize=3, alpha=0.4))
        ax.set_title(column, fontsize=10)
        ax.set_xticklabels([])
    fig.suptitle("Figure 2  -  Boxplots of all variables (red points mark outliers)",
                 fontsize=14, fontweight="bold")
    save_figure("fig02_boxplots.png")


def _plot_target_ceiling(frame: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(frame[TARGET], bins=50, color="#4C72B0", edgecolor="white", linewidth=0.4)
    ax.axvline(frame[TARGET].max(), color="red", ls="--", lw=2,
               label=f"capping value = {frame[TARGET].max():.4f}")
    ax.set_xlabel("Median house value (USD 100,000)")
    ax.set_ylabel("Frequency")
    ax.set_title("Figure 3  -  Target distribution showing the ceiling at 5.0",
                 fontweight="bold")
    ax.legend()
    save_figure("fig03_target_ceiling.png")


def _plot_geospatial(frame: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    value = axes[0].scatter(frame["Longitude"], frame["Latitude"], c=frame[TARGET],
                            cmap="viridis", s=3, alpha=0.6, linewidths=0)
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    axes[0].set_title("California coastline - blocks coloured by median value")
    fig.colorbar(value, ax=axes[0], label="Median house value")
    income = axes[1].scatter(frame["Longitude"], frame["Latitude"], c=frame["MedInc"],
                             cmap="inferno", s=3, alpha=0.6, linewidths=0)
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    axes[1].set_title("Median income - strong north-south gradient")
    fig.colorbar(income, ax=axes[1], label="Median income (USD 10,000s)")
    fig.suptitle("Figure 4  -  Geospatial distribution of the housing blocks",
                 fontsize=14, fontweight="bold")
    save_figure("fig04_geospatial.png")


def _plot_missing(frame: pd.DataFrame, core: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.bar(core, frame[core].isna().sum().values, color="#C44E52")
    ax.set_title("Figure 5  -  Missing values per column (all zero)")
    ax.set_ylabel("Count of missing")
    plt.xticks(rotation=45)
    save_figure("fig05_missing_values.png")


def _target_bands(frame: pd.DataFrame, results: dict) -> None:
    bins = [0, 1.5, 3.0, 4.0, 5.01]
    labels = ["Low (<1.5)", "Moderate (1.5-3)", "High (3-4)", "Very high (>4)"]
    counts = pd.cut(frame[TARGET], bins=bins, labels=labels).value_counts().reindex(labels)
    table = pd.DataFrame({"Value band": labels,
                          "Count": counts.values,
                          "Percentage": (counts.values / len(frame) * 100).round(2)})
    subsection("1.9 Target class distribution (bands in USD 100,000)")
    LOGGER.info("%s", table.to_string(index=False))
    LOGGER.info("")
    LOGGER.info("  COMMENT: %.1f%% of blocks sit at or above 3.0 and a large cluster"
                % ((counts.values[2] + counts.values[3]) / len(frame) * 100))
    LOGGER.info("  piles up exactly at the 5.0005 ceiling, so any model is pulled")
    LOGGER.info("  towards predicting the cap for the most expensive blocks.")
    table.to_csv(TAB_DIR / "04_target_bands.csv", index=False)
    results["target_bands"] = table

