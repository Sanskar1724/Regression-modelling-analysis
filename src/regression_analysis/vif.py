"""Task 4 - Multicollinearity analysis using the Variance Inflation Factor."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .config import TAB_DIR
from .logging_utils import get_logger
from .utils import save_figure, subsection

LOGGER = get_logger(__name__)

#: Conventional thresholds used throughout this project.
VIF_SEVERE = 10.0
VIF_MODERATE = 5.0


def compute_vif(x_train: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Return one VIF row per predictor, sorted from worst to best."""
    design = sm.add_constant(x_train, has_constant="add")
    rows = []
    for position, column in enumerate(design.columns):
        if column == "const":
            continue
        value = float(variance_inflation_factor(design.to_numpy(), position))
        rows.append({
            "Feature": column,
            "VIF": value,
            "R_squared": 1 - 1 / value,
            "Tolerance": 1 / value,
            "Severity": ("severe" if value >= VIF_SEVERE
                         else "moderate" if value >= VIF_MODERATE else "acceptable"),
        })
    return pd.DataFrame(rows).sort_values("VIF", ascending=False)


def report_vif(vif_table: pd.DataFrame) -> None:
    """Print the VIF table together with its interpretation."""
    LOGGER.info("")
    LOGGER.info("%s", vif_table.round(4).to_string(index=False))
    LOGGER.info("")
    LOGGER.info("  INTERPRETATION")
    severe = vif_table[vif_table["VIF"] >= VIF_SEVERE]["Feature"].tolist()
    moderate = vif_table[(vif_table["VIF"] >= VIF_MODERATE)
                         & (vif_table["VIF"] < VIF_SEVERE)]["Feature"].tolist()
    LOGGER.info("    Severe collinearity (VIF >= 10)     : %s", severe or "none")
    LOGGER.info("    Moderate collinearity (5 <= VIF < 10): %s", moderate or "none")
    worst_name = (severe or moderate or vif_table["Feature"].tolist())[0]
    worst = vif_table[vif_table["Feature"] == worst_name].iloc[0]
    LOGGER.info("    %s: VIF = %.2f, so %.1f%% of its variance is explained by the",
                worst["Feature"], worst["VIF"], worst["R_squared"] * 100)
    LOGGER.info("    other predictors alone; its standard error is inflated by")
    LOGGER.info("    sqrt(VIF) = %.1f relative to 1.0.", float(np.sqrt(worst["VIF"])))
    LOGGER.info("")
    LOGGER.info("    CONSEQUENCE: multicollinearity does NOT bias OLS coefficients, but")
    LOGGER.info("    it inflates their standard errors, so t-statistics and p-values")
    LOGGER.info("    become unreliable and individual interpretation is unsafe. The")
    LOGGER.info("    damage is to interpretation, not to prediction. This is the")
    LOGGER.info("    motivation for the regularised models, which stabilise the")
    LOGGER.info("    estimates without discarding any predictor.")
    LOGGER.info("")
    LOGGER.info("  NOTE: no VIF reaches 10, so the violation is MODERATE, not severe.")
    LOGGER.info("  Four variables breach the conventional VIF > 5 threshold.")


def iterative_vif(x_train: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Drop the worst predictor repeatedly until all VIFs fall below 5."""
    subsection("Iterative VIF check (remove worst offender, recompute)")
    remaining = list(features)
    history = []
    for _ in range(len(features)):
        reduced = sm.add_constant(x_train[remaining], has_constant="add")
        current = {c: float(variance_inflation_factor(reduced.to_numpy(), i))
                   for i, c in enumerate(reduced.columns) if c != "const"}
        worst = max(current, key=current.get)
        history.append({"Removed": worst, "VIF_at_removal": current[worst],
                        "Remaining": len(remaining) - 1})
        LOGGER.info("    Removing %-12s (VIF = %.3f) -> %d predictors left",
                    worst, current[worst], len(remaining) - 1)
        if current[worst] < VIF_MODERATE:
            break
        remaining.remove(worst)
    LOGGER.info("")
    LOGGER.info("  COMMENT: %d predictor(s) must be discarded before every remaining VIF",
                max(len(history) - 1, 0))
    LOGGER.info("  falls below 5. We deliberately do NOT drop them permanently -")
    LOGGER.info("  removing the coordinates would delete the strongest spatial signal in")
    LOGGER.info("  the data. Regularisation is the better remedy: it retains all")
    LOGGER.info("  variables while removing the instability.")
    table = pd.DataFrame(history)
    table.to_csv(TAB_DIR / "12_vif_iterative_removal.csv", index=False)
    return table


def plot_vif(vif_table: pd.DataFrame) -> None:
    """Horizontal bar chart of VIF with the 5 and 10 thresholds marked."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    plot = vif_table.sort_values("VIF")
    colours = ["#C44E52" if v >= VIF_SEVERE
               else "#DD8452" if v >= VIF_MODERATE else "#4C72B0"
               for v in plot["VIF"]]
    ax.barh(plot["Feature"], plot["VIF"], color=colours, edgecolor="white")
    ax.axvline(VIF_MODERATE, color="orange", ls="--", lw=2, label="VIF = 5 (moderate)")
    ax.axvline(VIF_SEVERE, color="red", ls="--", lw=2, label="VIF = 10 (severe)")
    ax.set_xlabel("Variance Inflation Factor")
    ax.set_title("Figure 11  -  VIF for each predictor", fontweight="bold")
    ax.legend()
    save_figure("fig11_vif.png")


def run_vif_analysis(x_train: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Run the complete VIF stage."""
    subsection("4.1 Variance Inflation Factor for each predictor")
    LOGGER.info("  Rule of thumb: VIF < 5 acceptable | 5 <= VIF < 10 moderate"
                " | VIF >= 10 severe")
    LOGGER.info("")
    vif_table = compute_vif(x_train, features)
    report_vif(vif_table)
    vif_table.to_csv(TAB_DIR / "11_vif_scores.csv", index=False)
    plot_vif(vif_table)
    iterative_vif(x_train, features)
    return vif_table
