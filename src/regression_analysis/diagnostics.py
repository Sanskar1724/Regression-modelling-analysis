"""Task 2 - correlation analysis.

Produces the full Pearson matrix, the rank-ordered correlation of each
predictor with the target, and the predictor-predictor pairs that foreshadow
the multicollinearity measured in ``vif.py``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import TAB_DIR, TARGET
from .logging_utils import get_logger
from .utils import save_figure, subsection

LOGGER = get_logger(__name__)


def correlation_analysis(frame: pd.DataFrame, features: list[str]) -> dict:
    """Pearson correlation matrix, target correlations and collinear pairs."""
    results: dict[str, pd.DataFrame] = {}
    core = features + [TARGET]

    subsection("2.1 Pearson correlation matrix")
    corr = frame[core].corr(method="pearson")
    LOGGER.info("%s", corr.round(4).to_string())
    corr.to_csv(TAB_DIR / "05_pearson_correlation_matrix.csv")
    results["matrix"] = corr

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Figure 6  -  Pearson correlation heatmap", fontweight="bold")
    save_figure("fig06_correlation_heatmap.png")

    subsection("2.2 Correlation of each predictor with the target")
    target_corr = corr[TARGET].drop(TARGET)
    ranked = pd.DataFrame({
        "Pearson_r": target_corr,
        "R_squared": (target_corr ** 2).round(4),
        "abs_rank": target_corr.abs().rank(ascending=False).astype(int),
        "strength": np.where(target_corr.abs() >= 0.7, "strong",
                             np.where(target_corr.abs() >= 0.4, "moderate", "weak")),
    }).sort_values("Pearson_r", key=abs, ascending=False)
    LOGGER.info("%s", ranked.round(4).to_string())
    LOGGER.info("")
    LOGGER.info("  COMMENT: MedInc correlates %.3f with the target, so income alone"
                % ranked.loc["MedInc", "Pearson_r"])
    LOGGER.info("  explains about %.0f%% of the variance - by far the strongest single"
                % (ranked.loc["MedInc", "R_squared"] * 100))
    LOGGER.info("  driver. Every other predictor has |r| below 0.16.")
    LOGGER.info("")
    LOGGER.info("  IMPORTANT: correlation is a MARGINAL measure. The near-zero values")
    LOGGER.info("  for AveBedrms, Population and AveOccup do NOT mean those variables")
    LOGGER.info("  are useless - the coefficient analysis later shows AveBedrms has the")
    LOGGER.info("  fourth largest coefficient once the other variables are controlled.")
    ranked.to_csv(TAB_DIR / "06_feature_target_correlation.csv")
    results["target"] = ranked

    subsection("2.3 Predictor-predictor correlation (multicollinearity preview)")
    predictor_corr = corr.loc[features, features]
    pairs = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            r = predictor_corr.iloc[i, j]
            pairs.append({
                "Feature_1": features[i], "Feature_2": features[j],
                "Pearson_r": r, "abs_r": abs(r),
                "flag": "HIGH" if abs(r) > 0.7 else ("moderate" if abs(r) > 0.5 else "low"),
            })
    pair_table = pd.DataFrame(pairs).sort_values("abs_r", ascending=False)
    LOGGER.info("%s", pair_table.round(4).to_string(index=False))
    high = pair_table[pair_table["flag"] == "HIGH"]
    LOGGER.info("")
    LOGGER.info("  COMMENT: %d predictor pair(s) exceed |r| = 0.7:", len(high))
    for _, row in high.iterrows():
        LOGGER.info("      %-10s <-> %-10s  r = %+.4f",
                    row["Feature_1"], row["Feature_2"], row["Pearson_r"])
    LOGGER.info("  This is the first warning sign of multicollinearity, confirmed")
    LOGGER.info("  quantitatively by the VIF analysis below.")
    pair_table.to_csv(TAB_DIR / "07_predictor_pair_correlation.csv", index=False)
    results["pairs"] = pair_table

    _plot_scatter(frame, features)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(predictor_corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax)
    ax.set_title("Figure 8  -  Predictor-only correlation matrix (collinearity check)",
                 fontweight="bold")
    save_figure("fig08_predictor_correlation.png")
    return results


def _plot_scatter(frame: pd.DataFrame, features: list[str]) -> None:
    ncols = 4
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.4 * nrows))
    for ax, feature in zip(axes.flat, features):
        ax.scatter(frame[feature], frame[TARGET], s=6, alpha=0.35,
                   color="#4C72B0", edgecolors="none")
        slope, intercept = np.polyfit(frame[feature], frame[TARGET], 1)
        xs = np.linspace(frame[feature].min(), frame[feature].max(), 100)
        ax.plot(xs, slope * xs + intercept, color="red", lw=2)
        r = frame[[feature, TARGET]].corr().iloc[0, 1]
        ax.set_title(f"{feature} vs {TARGET}\nr = {r:.3f}, R2 = {r ** 2:.3f}", fontsize=9)
        ax.set_xlabel(feature)
        ax.set_ylabel(TARGET)
    for ax in axes.flat[len(features):]:
        ax.axis("off")
    fig.suptitle("Figure 7  -  Scatter plot of every predictor against the target",
                 fontsize=14, fontweight="bold")
    save_figure("fig07_scatter_vs_target.png")
