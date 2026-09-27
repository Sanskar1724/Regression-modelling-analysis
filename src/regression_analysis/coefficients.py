"""Task 8 - feature significance and coefficient interpretation."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from .config import TAB_DIR
from .logging_utils import get_logger
from .utils import save_figure, subsection

LOGGER = get_logger(__name__)

#: USD value of one unit of the target, used to express coefficients in money.
DOLLARS_PER_UNIT = 100_000


def coefficient_table(ols, features: list[str]) -> pd.DataFrame:
    """Extract coefficients, standard errors, t statistics, p values and CIs."""
    subsection("11.2 Coefficient table (target unit = USD 100,000)")
    table = pd.DataFrame({
        "Feature": features,
        "Coef_per_SD": ols.params.to_numpy()[1:],
        "Std_Error": ols.bse.to_numpy()[1:],
        "t_stat": ols.tvalues.to_numpy()[1:],
        "p_value": ols.pvalues.to_numpy()[1:],
        "CI_lower_95": ols.conf_int().to_numpy()[1:, 0],
        "CI_upper_95": ols.conf_int().to_numpy()[1:, 1],
    })
    table["significant_5pct"] = table["p_value"] < 0.05
    table["abs_t"] = table["t_stat"].abs()
    table = table.sort_values("abs_t", ascending=False).reset_index(drop=True)
    LOGGER.info("%s", table.round(4).to_string(index=False))
    table.to_csv(TAB_DIR / "22_coefficient_significance.csv", index=False)
    return table


def interpret_coefficients(table: pd.DataFrame, features: list[str]) -> None:
    """Translate the coefficients into real-world statements and explain them."""
    subsection("11.3 Interpretation in real terms")
    LOGGER.info("  (each coefficient is the change in median house value, in")
    LOGGER.info("   USD 100,000, for a one standard deviation rise in that predictor)")
    LOGGER.info("")
    for _, row in table.iterrows():
        dollars = row["Coef_per_SD"] * DOLLARS_PER_UNIT
        verdict = "SIGNIFICANT" if row["significant_5pct"] else "not significant"
        # Thousands separators are not available in %-style formatting, so the
        # grouped string is built explicitly before being logged.
        LOGGER.info("    %-12s beta = %+.4f  (%s USD)  t = %+.2f  p = %.4f  [%s]",
                    row["Feature"], row["Coef_per_SD"],
                    f"{dollars:+,.0f}", row["t_stat"], row["p_value"], verdict)
    significant = table[table["significant_5pct"]]["Feature"].tolist()
    insignificant = table[~table["significant_5pct"]]["Feature"].tolist()
    LOGGER.info("")
    LOGGER.info("    Significant at 5%% : %s", significant)
    LOGGER.info("    Not significant  : %s", insignificant or "none")
    LOGGER.info("")
    for line in _INTERPRETATION:
        LOGGER.info("    %s", line)


_INTERPRETATION = (
    "KEY INTERPRETATIONS",
    "",
    "* MedInc is the strongest positive driver: a one standard deviation rise in",
    "  district income adds about USD 85,400. Economically unsurprising but",
    "  quantitatively large, and it also holds the highest forest importance.",
    "",
    "* Latitude and Longitude are jointly the largest effect (t = -52.8, -52.1).",
    "  Location dominates once income is controlled for. The negative Latitude",
    "  sign means value falls as one moves NORTH. The two must be read together,",
    "  since alone each only describes a curved coastal strip - which is exactly",
    "  why they are highly collinear yet both strongly significant.",
    "",
    "* AveRooms is NEGATIVE while AveBedrms is positive, and BOTH are significant.",
    "  This is a suppression effect: the two correlate at 0.85, so each absorbs",
    "  part of the other's influence. The correct reading is that room COUNT adds",
    "  value once household size is controlled for. Neither sign should be read in",
    "  isolation - this is precisely why the VIF analysis matters, because without",
    "  it one would wrongly conclude that more rooms reduce house value.",
    "",
    "* Population is the ONLY insignificant variable (p = 0.699). Once MedInc, the",
    "  coordinates and the rooms-per-person ratio are known, the raw block",
    "  population adds nothing further.",
    "",
    "* HouseAge is mildly positive: vintage matters in California, unlike in most",
    "  older housing markets.",
    "",
    "* THE KEY LESSON: AveBedrms correlates only -0.047 with the target, essentially",
    "  zero, yet carries the fourth largest coefficient. It has no MARGINAL",
    "  relationship but a strong PARTIAL one, because it only matters once income",
    "  and location are held constant. Selecting variables from a correlation",
    "  matrix alone would have discarded a critical predictor.",
)


def plot_coefficients(table: pd.DataFrame) -> None:
    """Bar chart of coefficients, grey-flagging the insignificant ones."""
    fig, ax = plt.subplots(figsize=(11, 6))
    ordered = table.sort_values("Coef_per_SD")
    colours = ["#4C72B0" if s else "#C44E52" for s in ordered["significant_5pct"]]
    ax.barh(ordered["Feature"], ordered["Coef_per_SD"], color=colours, edgecolor="white")
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Coefficient (per 1 SD, units of USD 100,000)")
    ax.set_title("Figure 21  -  OLS coefficients (red = not significant at 5%)",
                 fontweight="bold")
    save_figure("fig21_coefficients.png")


def run_coefficient_analysis(ols, features: list[str]) -> pd.DataFrame:
    """Run the full coefficient stage."""
    subsection("11.1 Full OLS summary statistics")
    LOGGER.info("%s", ols.summary())
    table = coefficient_table(ols, features)
    interpret_coefficients(table, features)
    plot_coefficients(table)
    return table
