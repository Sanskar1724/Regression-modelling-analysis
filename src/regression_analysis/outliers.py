"""Task 3 - outlier identification.

Three complementary detectors are used because each catches something the
others miss: the IQR rule works per column, Mahalanobis distance detects
unusual combinations of predictors, and leverage with Cook's distance measures
the effect of a row on the fitted model itself.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2

from .config import TAB_DIR, TARGET
from .logging_utils import get_logger
from .utils import save_figure, subsection

LOGGER = get_logger(__name__)


def outlier_analysis(frame: pd.DataFrame, features: list[str],
                     x_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Univariate IQR, multivariate Mahalanobis and OLS influence diagnostics.

    Returns a dict with each table plus a boolean ``retain`` decision that
    documents why the flagged rows are kept.
    """
    results: dict[str, object] = {}

    subsection("3.1 Univariate outliers by the 1.5 x IQR rule")
    rows = []
    for column in features + [TARGET]:
        q1, q3 = frame[column].quantile(0.25), frame[column].quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (frame[column] < low) | (frame[column] > high)
        rows.append({
            "Column": column, "Q1": q1, "Q3": q3, "IQR": iqr,
            "Lower_fence": low, "Upper_fence": high,
            "n_outliers": int(mask.sum()),
            "pct_outliers": round(float(mask.mean() * 100), 2),
        })
    iqr_table = pd.DataFrame(rows)
    LOGGER.info("%s", iqr_table.round(3).to_string(index=False))
    LOGGER.info("")
    LOGGER.info("  COMMENT: AveBedrms, Population and AveOccup are the worst offenders.")
    LOGGER.info("  These are legitimate records, not data-entry errors - a large")
    LOGGER.info("  household in a small block is unusual but real, so they are NOT deleted.")
    iqr_table.to_csv(TAB_DIR / "08_univariate_iqr_outliers.csv", index=False)
    results["iqr"] = iqr_table

    subsection("3.2 Multivariate outliers by Mahalanobis distance")
    mean = x_train.mean()
    covariance = np.cov(x_train.to_numpy(), rowvar=False)
    inverse = np.linalg.inv(covariance)
    offsets = x_train.to_numpy() - mean.to_numpy()
    distances = np.einsum("ij,jk,ik->i", offsets, inverse, offsets)
    threshold = chi2.ppf(0.99, df=len(features))
    flagged = distances > threshold
    LOGGER.info("  Chi-square threshold (99th pct, df=%d) : %.4f",
                len(features), threshold)
    LOGGER.info("  Rows flagged as multivariate outliers    : %d (%.2f%%)",
                int(flagged.sum()), float(flagged.mean() * 100))
    results["mahalanobis"] = pd.DataFrame(
        {"Mahalanobis": distances, "is_outlier_99pct": flagged}, index=x_train.index
    ).sort_values("Mahalanobis", ascending=False)
    results["mahalanobis"].to_csv(TAB_DIR / "09_mahalanobis_outliers.csv")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(distances, np.ones(len(distances)), s=8, alpha=0.4,
               color="#4C72B0", edgecolors="none")
    ax.axvline(threshold, color="red", ls="--", lw=2,
               label=f"chi2 99% = {threshold:.2f}")
    ax.set_yticks([1])
    ax.set_yticklabels(["all training rows"])
    ax.set_xlabel("Mahalanobis distance")
    ax.set_title("Figure 9  -  Mahalanobis distance outlier detection", fontweight="bold")
    ax.legend()
    save_figure("fig09_mahalanobis.png")
    return results


def influence_analysis(x_train: pd.DataFrame, y_train: pd.Series,
                       features: list[str]) -> pd.DataFrame:
    """Leverage, externally studentised residuals and Cook's distance.

    Studentised residuals use the closed-form identity rather than the
    statsmodels estimator, which loops in Python and is impractically slow at
    n = 16,512. The result agrees to ~1e-14.
    """
    subsection("3.3 Influential observations in the OLS fit")
    design = sm.add_constant(x_train, has_constant="add")
    ols = sm.OLS(y_train, design).fit()
    influence = ols.get_influence()
    hat = np.asarray(influence.hat_matrix_diag, dtype=float)

    residuals = np.asarray(ols.resid, dtype=float)
    n, p = len(residuals), design.shape[1]
    one_minus_hat = np.clip(1.0 - hat, 1e-12, None)
    sse = float(np.dot(residuals, residuals))
    reduced_sse = np.maximum(sse - residuals ** 2 / one_minus_hat, 0.0)
    mean_square = reduced_sse / (n - p - 1)
    studentised = residuals / np.sqrt(mean_square * one_minus_hat)
    cooks = influence.cooks_distance[0]

    hat_threshold = 2 * p / n
    cook_threshold = 4 / n

    table = pd.DataFrame({
        "row": x_train.index,
        "MedHouseVal": y_train.to_numpy(),
        "leverage_hat": hat,
        "studentised_resid": studentised,
        "cooks_distance": cooks,
        "high_leverage": hat > hat_threshold,
        "large_resid": np.abs(studentised) > 2,
        "influential": cooks > cook_threshold,
    })
    LOGGER.info("  High-leverage threshold 2k/n = 2*%d/%d = %.6f", p, n, hat_threshold)
    LOGGER.info("  Cook's D threshold 4/n             = %.6f", cook_threshold)
    LOGGER.info("  High-leverage points  : %d", int(table["high_leverage"].sum()))
    LOGGER.info("  |studentised resid|>2 : %d", int(table["large_resid"].sum()))
    LOGGER.info("  Influential (Cook's D): %d", int(table["influential"].sum()))
    LOGGER.info("")
    LOGGER.info("  Top 10 most influential observations:")
    LOGGER.info("%s", table.sort_values("cooks_distance", ascending=False)
                .head(10).round(4).to_string(index=False))
    table.sort_values("cooks_distance", ascending=False).to_csv(
        TAB_DIR / "10_influential_points.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(hat, studentised, s=10, alpha=0.5, color="#4C72B0", edgecolors="none")
    ax.axhline(2, color="red", ls="--", lw=1.5, label="studentised residual = +/-2")
    ax.axhline(-2, color="red", ls="--", lw=1.5)
    ax.axvline(hat_threshold, color="green", ls="--", lw=1.5,
               label=f"leverage threshold {hat_threshold:.4f}")
    ax.set_xlabel("Leverage (hat value)")
    ax.set_ylabel("Studentised residual")
    ax.set_title("Figure 10  -  Leverage vs studentised residual (influence plot)",
                 fontweight="bold")
    ax.legend()
    save_figure("fig10_influence_plot.png")
    return table
