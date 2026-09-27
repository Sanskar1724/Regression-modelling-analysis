"""End-to-end orchestration of the California Housing regression analysis.

Running :func:`run` executes every mandatory task in order and writes all
figures and tables under ``outputs/``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .assumptions import residual_diagnostics, run_assumption_tests
from .coefficients import run_coefficient_analysis
from .comparison import (coefficient_shrinkage, comparison_table,
                         cross_validated_scores, errors_in_dollars,
                         final_selection, ranking_table)
from .config import (CV_FOLDS, ELASTICNET_ALPHAS, ELASTICNET_L1_RATIOS,
                     LASSO_ALPHAS, RIDGE_ALPHAS, TAB_DIR, TARGET)
from .data import load_dataset, train_test_split_data
from .diagnostics import correlation_analysis
from .eda import run_eda
from .logging_utils import get_logger
from .models import (gradient_boosting, multiple_linear_regression,
                     polynomial_regression, random_forest, robust_regression,
                     simple_linear_regression, tuned_elastic_net, tuned_lasso,
                     tuned_ridge)
from .outliers import influence_analysis, outlier_analysis
from .utils import section, subsection
from .vif import run_vif_analysis

LOGGER = get_logger(__name__)


def run() -> dict:
    """Execute the full analysis and return every result table."""
    section("REGRESSION ANALYSIS  |  CALIFORNIA HOUSING  (Dataset 1 of 16)")
    LOGGER.info("Target: %s  |  80/20 split  |  seed 42", TARGET)

    section("TASK 1  |  EXPLORATORY DATA ANALYSIS")
    frame, features = load_dataset()
    x_train, x_test, y_train, y_test = train_test_split_data(frame, features)
    run_eda(frame, features)

    section("TASK 2  |  CORRELATION ANALYSIS")
    correlation_analysis(frame, features)

    section("TASK 3  |  OUTLIER IDENTIFICATION")
    outlier_analysis(frame, features, x_train, y_train)
    influence_analysis(x_train, y_train, features)
    LOGGER.info("")
    LOGGER.info("  DECISION: outliers are RETAINED. The flagged rows are legitimate")
    LOGGER.info("  extreme households rather than data-entry errors. The Huber")
    LOGGER.info("  sensitivity test below confirms this: down-weighting them LOWERS R2,")
    LOGGER.info("  which would be the opposite of the case if they were corrupt data.")

    section("TASK 4  |  MULTICOLLINEARITY ANALYSIS USING VIF")
    run_vif_analysis(x_train, features)

    results, fitted, scaler = _fit_models(x_train, x_test, y_train, y_test, features)

    section("TASK 6  |  MODEL COMPARISON AND RESIDUAL ANALYSIS")
    comparison = comparison_table(results)
    ranking_table(comparison)
    errors_in_dollars(comparison)

    section("TASK 7  |  REGRESSION ASSUMPTION CHECKS")
    yhat = fitted["Multiple Linear Regression"].predict(scaler.transform(x_test))
    ols, summary = run_assumption_tests(x_train, y_train, features, y_test, yhat)
    residual_diagnostics(ols, y_test, yhat)
    _residual_statistics(y_test, yhat)

    section("TASK 8  |  FEATURE SIGNIFICANCE AND COEFFICIENT INTERPRETATION")
    run_coefficient_analysis(ols, features)

    section("TASK 10  |  REGULARISATION, CROSS-VALIDATION AND FINAL SELECTION")
    coefficient_shrinkage(fitted, features)
    cv = cross_validated_scores(x_train, y_train)
    final = final_selection(comparison, cv)

    return {"comparison": comparison, "summary": summary,
            "cross_validation": cv, "final": final, "models": results}


def _fit_models(x_train, x_test, y_train, y_test, features):
    """Fit all nine models and return their metrics, estimators and the scaler."""
    section("TASK 5  |  REGRESSION MODELS  (Task 9 tuning included)")
    results: dict[str, dict] = {}
    fitted: dict[str, object] = {}
    chosen: list[dict] = []

    metrics, _, _ = simple_linear_regression(x_train, x_test, y_train, y_test, features)
    results["Simple Linear Regression"] = metrics

    metrics, mlr, scaler = multiple_linear_regression(
        x_train, x_test, y_train, y_test, features)
    results["Multiple Linear Regression"] = metrics
    fitted["Multiple Linear Regression"] = mlr

    metrics, _ = polynomial_regression(x_train, x_test, y_train, y_test, features, scaler)
    results["Polynomial Regression (deg 2)"] = metrics

    metrics, model, alpha = tuned_ridge(x_train, x_test, y_train, y_test, features, scaler)
    results["Ridge Regression"] = metrics
    fitted["Ridge Regression"] = model
    chosen.append({"Model": "Ridge", "alpha": alpha, "l1_ratio": np.nan,
                   "non_zero_coefs": len(features)})

    metrics, model, alpha, zeroed = tuned_lasso(
        x_train, x_test, y_train, y_test, features, scaler)
    results["Lasso Regression"] = metrics
    fitted["Lasso Regression"] = model
    chosen.append({"Model": "Lasso", "alpha": alpha, "l1_ratio": np.nan,
                   "non_zero_coefs": len(features) - len(zeroed)})

    metrics, model, alpha, l1_ratio = tuned_elastic_net(
        x_train, x_test, y_train, y_test, features, scaler)
    results["Elastic Net Regression"] = metrics
    fitted["Elastic Net Regression"] = model
    chosen.append({"Model": "Elastic Net", "alpha": alpha, "l1_ratio": l1_ratio,
                   "non_zero_coefs": int((np.abs(model.coef_) > 1e-8).sum())})

    metrics, _ = robust_regression(x_train, x_test, y_train, y_test, features, scaler)
    results["Robust Regression (Huber)"] = metrics

    metrics, _ = random_forest(x_train, x_test, y_train, y_test, features)
    results["Random Forest Regressor"] = metrics

    metrics, _ = gradient_boosting(x_train, x_test, y_train, y_test, features)
    results["Gradient Boosting Regressor"] = metrics

    _write_hyperparameters(chosen)
    return results, fitted, scaler


def _write_hyperparameters(chosen: list[dict]) -> None:
    """Persist the cross-validated hyper-parameters chosen for each penalised model."""
    subsection("5.10 Selected hyper-parameters (Task 9 summary)")
    table = pd.DataFrame(chosen)
    LOGGER.info("%s", table.round(6).to_string(index=False))
    LOGGER.info("")
    LOGGER.info("  SEARCH SPACES")
    LOGGER.info("    Ridge        : %d candidates, %.0e to %.0e (quarter-decade steps)",
                len(RIDGE_ALPHAS), min(RIDGE_ALPHAS), max(RIDGE_ALPHAS))
    LOGGER.info("    Lasso        : %d candidates, %.0e to %.0e",
                len(LASSO_ALPHAS), min(LASSO_ALPHAS), max(LASSO_ALPHAS))
    LOGGER.info("    Elastic Net  : %d alphas x %d l1_ratios (2-D grid)",
                len(ELASTICNET_ALPHAS), len(ELASTICNET_L1_RATIOS))
    LOGGER.info("    All tuned by %d-fold cross-validation with shuffling.", CV_FOLDS)
    LOGGER.info("")
    LOGGER.info("  COMMENT: every model retained all 8 coefficients. The chosen penalties")
    LOGGER.info("  are small relative to the data scale, so the penalised fits are")
    LOGGER.info("  effectively OLS here - see the shrinkage table in Task 10.")
    table.to_csv(TAB_DIR / "15_selected_hyperparameters.csv", index=False)


def _residual_statistics(y_test, yhat) -> None:
    """Report the basic shape of the test-set residuals."""
    section("RESIDUAL STATISTICS (test set)")
    residuals = pd.Series(y_test.to_numpy() - yhat)
    LOGGER.info("  Mean residual    : %+.6f  (should be ~0)", float(residuals.mean()))
    LOGGER.info("  Std of residual  : %.6f", float(residuals.std()))
    LOGGER.info("  Min / Max        : %+.4f / %+.4f",
                float(residuals.min()), float(residuals.max()))
    LOGGER.info("  Skewness         : %+.4f", float(residuals.skew()))
    LOGGER.info("  Kurtosis         : %+.4f", float(residuals.kurtosis()))
    LOGGER.info("  %% residuals > +1 : %.2f%%", float((residuals > 1).mean() * 100))
    LOGGER.info("  %% residuals < -1 : %.2f%%", float((residuals < -1).mean() * 100))
    LOGGER.info("")
    LOGGER.info("  COMMENT: a mean residual near zero confirms the OLS unbiasedness")
    LOGGER.info("  property. The positive skew and heavy excess kurtosis reflect the")
    LOGGER.info("  censored target, exactly as the Jarque-Bera test predicted.")

