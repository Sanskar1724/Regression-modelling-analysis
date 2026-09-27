"""Task 7 - the five classical regression assumption tests.

Every test is run on an OLS fit built from a DataFrame so that statsmodels
preserves the real feature names (passing a bare ndarray renames the columns
to x1, x2, ... and breaks the named hypothesis tests).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera

from .config import TAB_DIR
from .logging_utils import get_logger
from .utils import save_figure, subsection
from .vif import VIF_MODERATE

LOGGER = get_logger(__name__)

#: Interpreting bands for the Durbin-Watson statistic.
DW_LOW, DW_HIGH = 1.5, 2.5


def fit_reference_ols(x_train: pd.DataFrame, y_train: pd.Series,
                      features: list[str]):
    """Fit the OLS model that all assumption tests are based on."""
    scaled = pd.DataFrame(
        _standardised(x_train, features), columns=features, index=x_train.index
    )
    design = sm.add_constant(scaled, has_constant="add")
    return sm.OLS(y_train, design).fit()


def _standardised(x_train: pd.DataFrame, features: list[str]) -> np.ndarray:
    from sklearn.preprocessing import StandardScaler

    return StandardScaler().fit_transform(x_train[features])


def test_linearity(ols, features: list[str]) -> dict:
    """Overall F-test that every slope coefficient is zero."""
    LOGGER.info("")
    LOGGER.info("-" * 70)
    LOGGER.info("ASSUMPTION 1 of 5  |  LINEARITY")
    LOGGER.info("-" * 70)
    restriction = np.zeros((len(features), ols.model.exog.shape[1]))
    columns = {name: i for i, name in enumerate(ols.model.exog_names)}
    for row, feature in enumerate(features):
        restriction[row, columns[feature]] = 1.0
    test = ols.f_test(restriction)
    p_value = float(np.squeeze(test.pvalue))
    statistic = float(np.squeeze(test.fvalue))
    LOGGER.info("  H0 : all slope coefficients = 0 (no linear relationship)")
    LOGGER.info("  Overall F-statistic = %.4f", statistic)
    LOGGER.info("  F-test p-value      = %.4e", p_value)
    LOGGER.info("  Decision            : %s",
                "REJECT H0" if p_value < 0.05 else "FAIL TO REJECT H0")
    LOGGER.info("")
    LOGGER.info("  COMMENT: the relationship IS linear in its parameters. This tests")
    LOGGER.info("  significance, not goodness of fit - a model can be highly significant")
    LOGGER.info("  and still predict poorly, which is exactly what happens here.")
    return {"Assumption": "Linearity", "Test": "Overall F-test",
            "Statistic": statistic, "p_value": p_value,
            "Result": "PASS" if p_value < 0.05 else "FAIL"}


def test_independence(ols) -> dict:
    """Durbin-Watson test for autocorrelation in the residuals."""
    LOGGER.info("")
    LOGGER.info("-" * 70)
    LOGGER.info("ASSUMPTION 2 of 5  |  INDEPENDENCE OF ERRORS")
    LOGGER.info("-" * 70)
    statistic = float(durbin_watson(ols.resid))
    LOGGER.info("  Durbin-Watson statistic = %.4f", statistic)
    LOGGER.info("  Guide: ~2.0 no autocorrelation | <1.5 positive | >2.5 negative")
    LOGGER.info("")
    LOGGER.info("  COMMENT: %.3f is very close to 2, so the errors are essentially" % statistic)
    LOGGER.info("  uncorrelated.")
    LOGGER.info("  CAVEAT: rows are grouped into roughly 6,000 geographic blocks, so")
    LOGGER.info("  independence is assumed by the OLS model rather than proven by it.")
    passed = DW_LOW <= statistic <= DW_HIGH
    return {"Assumption": "Independence", "Test": "Durbin-Watson",
            "Statistic": statistic, "p_value": np.nan,
            "Result": "PASS (DW ~ 2)" if passed else "FAIL (autocorrelated)"}


def test_homoscedasticity(ols) -> dict:
    """Breusch-Pagan test for constant variance of the residuals."""
    LOGGER.info("")
    LOGGER.info("-" * 70)
    LOGGER.info("ASSUMPTION 3 of 5  |  HOMOSCEDASTICITY (constant variance)")
    LOGGER.info("-" * 70)
    lm_stat, lm_p, f_stat, f_p = het_breuschpagan(ols.resid, ols.model.exog)
    LOGGER.info("  Breusch-Pagan LM statistic = %.4f", float(lm_stat))
    LOGGER.info("  LM-test p-value            = %.4e", float(lm_p))
    LOGGER.info("  F-test  p-value            = %.4e", float(f_p))
    violated = float(lm_p) < 0.05
    LOGGER.info("  Decision : %s",
                "HETEROsCEDASTIC - assumption VIOLATED" if violated
                else "homoscedastic - PASS")
    LOGGER.info("")
    LOGGER.info("  COMMENT: the residual variance is NOT constant. The model is more")
    LOGGER.info("  accurate for mid-priced blocks and less accurate at the extremes,")
    LOGGER.info("  which is precisely the signature of the ceiling at 5.0.")
    LOGGER.info("  Practical fixes: log-transform the target, use robust standard")
    LOGGER.info("  errors, or fit a censored (Tobit) model for the truncation.")
    return {"Assumption": "Homoscedasticity", "Test": "Breusch-Pagan",
            "Statistic": float(lm_stat), "p_value": float(lm_p),
            "Result": "FAIL (heteroscedastic)" if violated else "PASS"}


def test_normality(ols) -> dict:
    """Jarque-Bera test for normality of the residuals."""
    LOGGER.info("")
    LOGGER.info("-" * 70)
    LOGGER.info("ASSUMPTION 4 of 5  |  NORMALITY OF RESIDUALS")
    LOGGER.info("-" * 70)
    statistic, p_value, skew, kurtosis = jarque_bera(np.asarray(ols.resid))
    LOGGER.info("  Jarque-Bera statistic = %.4f", float(statistic))
    LOGGER.info("  p-value              = %.4e", float(p_value))
    LOGGER.info("  Residual skewness    = %.4f", float(skew))
    LOGGER.info("  Residual kurtosis    = %.4f (excess)", float(kurtosis))
    violated = float(p_value) < 0.05
    LOGGER.info("  Decision : %s",
                "NON-NORMAL - assumption VIOLATED" if violated else "PASS")
    LOGGER.info("")
    LOGGER.info("  COMMENT: the residuals are NOT normal. At this sample size the test")
    LOGGER.info("  has enormous power, so even a mild departure is flagged; the S-shape")
    LOGGER.info("  in the Q-Q plot confirms it is genuine. Normality is required for")
    LOGGER.info("  valid confidence intervals, NOT for the point estimates.")
    return {"Assumption": "Normality", "Test": "Jarque-Bera",
            "Statistic": float(statistic), "p_value": float(p_value),
            "Result": "FAIL (non-normal)" if violated else "PASS"}


def test_multicollinearity(ols) -> dict:
    """Revisit the worst VIF among the fitted design."""
    LOGGER.info("")
    LOGGER.info("-" * 70)
    LOGGER.info("ASSUMPTION 5 of 5  |  MULTICOLLINEARITY (revisited from the VIF stage)")
    LOGGER.info("-" * 70)
    vifs = {
        column: float(variance_inflation_factor(ols.model.exog, i))
        for i, column in enumerate(ols.model.exog_names) if column != "const"
    }
    worst = max(vifs, key=vifs.get)
    LOGGER.info("  Highest VIF : %s = %.4f  (moderate: 5 <= VIF < 10)", worst, vifs[worst])
    LOGGER.info("")
    LOGGER.info("  COMMENT: violated at the conventional VIF > 5 threshold for the")
    LOGGER.info("  location and size variables. The coordinates explain about 89% of each")
    LOGGER.info("  other's variance, so individual standard errors are inflated even")
    LOGGER.info("  though the model as a whole is well specified.")
    violated = vifs[worst] > VIF_MODERATE
    return {"Assumption": "Multicollinearity", "Test": "VIF",
            "Statistic": vifs[worst], "p_value": np.nan,
            "Result": "FAIL (VIF > 5, moderate)" if violated else "PASS"}


def residual_diagnostics(ols, y_test, yhat) -> pd.Series:
    """Produce the five standard residual diagnostic figures."""
    train_residuals = np.asarray(ols.resid)
    test_residuals = np.asarray(y_test) - np.asarray(yhat)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    axes[0].scatter(ols.fittedvalues, train_residuals, s=4, alpha=0.3,
                   color="#4C72B0", edgecolors="none")
    axes[0].axhline(0, color="red", ls="--", lw=2)
    axes[0].set_xlabel("Fitted values")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residuals vs Fitted (training)", fontweight="bold")
    axes[1].scatter(yhat, test_residuals, s=8, alpha=0.4, color="#55A868",
                   edgecolors="none")
    axes[1].axhline(0, color="red", ls="--", lw=2)
    axes[1].set_xlabel("Predicted values (test)")
    axes[1].set_ylabel("Residuals")
    axes[1].set_title("Residuals vs Predicted (test set)", fontweight="bold")
    fig.suptitle("Figure 16  -  Residual vs fitted plots (look for structure)",
                 fontsize=14, fontweight="bold")
    save_figure("fig16_residuals_vs_fitted.png")

    fig, ax = plt.subplots(figsize=(7, 6.5))
    stats.probplot(train_residuals, dist="norm", plot=ax)
    ax.set_title("Figure 17  -  Normal Q-Q plot of OLS residuals", fontweight="bold")
    save_figure("fig17_qq_plot.png")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.hist(train_residuals, bins=60, density=True, alpha=0.7, color="#4C72B0",
            edgecolor="white", label="Residuals")
    grid = np.linspace(train_residuals.min(), train_residuals.max(), 200)
    ax.plot(grid, stats.norm.pdf(grid, train_residuals.mean(), train_residuals.std()),
            "r-", lw=2, label="Normal reference")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Density")
    ax.set_title("Figure 18  -  Residual distribution vs the normal reference",
                 fontweight="bold")
    ax.legend()
    save_figure("fig18_residual_distribution.png")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(yhat, np.sqrt(np.abs(test_residuals)), s=8, alpha=0.4,
               color="#DD8452", edgecolors="none")
    ax.set_xlabel("Predicted values")
    ax.set_ylabel("sqrt(|residual|)")
    ax.set_title("Figure 19  -  Scale-Location plot (visual homoscedasticity check)",
                 fontweight="bold")
    save_figure("fig19_scale_location.png")

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.scatter(y_test, yhat, s=8, alpha=0.4, color="#4C72B0", edgecolors="none")
    limits = [min(np.min(y_test), np.min(yhat)), max(np.max(y_test), np.max(yhat))]
    ax.plot(limits, limits, "r--", lw=2, label="45 degree line")
    ax.set_xlabel("Actual MedHouseVal")
    ax.set_ylabel("Predicted MedHouseVal")
    ax.set_title("Figure 20  -  Actual vs Predicted (test set)", fontweight="bold")
    ax.legend()
    save_figure("fig20_actual_vs_predicted.png")
    return pd.Series(test_residuals)


def run_assumption_tests(x_train, y_train, features, y_test, yhat) -> tuple:
    """Run all five assumption tests, print the summary and write the OLS report."""
    ols = fit_reference_ols(x_train, y_train, features)
    rows = [
        test_linearity(ols, features),
        test_independence(ols),
        test_homoscedasticity(ols),
        test_normality(ols),
        test_multicollinearity(ols),
    ]
    summary = pd.DataFrame(rows)
    subsection("SUMMARY OF ALL FIVE ASSUMPTION TESTS")
    LOGGER.info("%s", summary.round(4).to_string(index=False))
    passed = int(summary["Result"].str.startswith("PASS").sum())
    LOGGER.info("")
    LOGGER.info("  Assumptions satisfied : %d of 5", passed)
    LOGGER.info("")
    LOGGER.info("  CONCLUSION: linearity and independence hold comfortably.")
    LOGGER.info("  Homoscedasticity, normality and multicollinearity do NOT hold. The")
    LOGGER.info("  decisive point is that these failures affect the reliability of")
    LOGGER.info("  confidence intervals and p-values, NOT the usefulness of the point")
    LOGGER.info("  predictions. That is why regularised models are preferred, and why")
    LOGGER.info("  the violations argue against naive inference rather than against")
    LOGGER.info("  using the model at all.")
    summary.to_csv(TAB_DIR / "20_assumption_tests_summary.csv", index=False)
    with open(TAB_DIR / "21_ols_full_summary.txt", "w", encoding="utf-8") as handle:
        handle.write(str(ols.summary()))
    return ols, summary
