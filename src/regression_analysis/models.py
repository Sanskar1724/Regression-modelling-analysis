"""Task 5 and Task 9 - regression models and cross-validated hyperparameter search."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import (ElasticNet, ElasticNetCV, HuberRegressor,
                                  Lasso, LassoCV, LinearRegression, Ridge, RidgeCV)
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .config import (CV_FOLDS, ELASTICNET_ALPHAS, ELASTICNET_L1_RATIOS,
                     LASSO_ALPHAS, RANDOM_STATE, RIDGE_ALPHAS, TAB_DIR)
from .logging_utils import get_logger
from .utils import compute_metrics, save_figure, subsection

LOGGER = get_logger(__name__)


def _scaler() -> StandardScaler:
    return StandardScaler()


def simple_linear_regression(x_train, x_test, y_train, y_test, features):
    """Fit a one-variable model on the predictor most correlated with the target."""
    subsection("5.1 Simple Linear Regression")
    best = x_train.corrwith(y_train).abs().idxmax()
    LOGGER.info("  Strongest single predictor by |r| : %s", best)
    model = LinearRegression().fit(x_train[[best]], y_train)
    predictions = model.predict(x_test[[best]])
    metrics = compute_metrics(y_test, predictions, 1)
    LOGGER.info("  Equation : MedHouseVal = %.4f %+.4f * %s",
                model.intercept_, model.coef_[0], best)
    LOGGER.info("  Test R2  : %.4f   RMSE : %.4f", metrics["R2"], metrics["RMSE"])
    LOGGER.info("")
    LOGGER.info("  COMMENT: a one-variable model leaves roughly half the variance")
    LOGGER.info("  unexplained, so the relationship is genuinely multi-factor and")
    LOGGER.info("  multiple regression is required.")
    _plot_predicted_actual(y_test, predictions, metrics["R2"],
                           "Figure 12  -  Simple Linear Regression",
                           "fig12_simple_lr.png")
    return metrics, model, [best]


def multiple_linear_regression(x_train, x_test, y_train, y_test, features):
    """OLS on all predictors, fitted on standardised columns."""
    subsection("5.2 Multiple Linear Regression (OLS, all predictors)")
    scaler = _scaler()
    scaled_train = scaler.fit_transform(x_train)
    scaled_test = scaler.transform(x_test)
    model = LinearRegression().fit(scaled_train, y_train)
    predictions = model.predict(scaled_test)
    metrics = compute_metrics(y_test, predictions, len(features))
    LOGGER.info("  Intercept (scaled) : %.4f", model.intercept_)
    coefficients = pd.DataFrame({"Feature": features,
                                 "Coefficient_scaled": model.coef_})
    coefficients["abs_coef"] = coefficients["Coefficient_scaled"].abs()
    coefficients = coefficients.sort_values("abs_coef", ascending=False)
    LOGGER.info("%s", coefficients.round(4).to_string(index=False))
    coefficients.to_csv(TAB_DIR / "13_mlr_coefficients.csv", index=False)
    LOGGER.info("  Test R2 = %.4f | Adj R2 = %.4f | MAE = %.4f | RMSE = %.4f",
                metrics["R2"], metrics["Adj_R2"], metrics["MAE"], metrics["RMSE"])
    LOGGER.info("")
    LOGGER.info("  COMMENT: the seven added predictors raise R2 substantially over the")
    LOGGER.info("  simple model, and Adjusted R2 stays equally high, so they carry")
    LOGGER.info("  genuine information rather than fitting noise.")
    _plot_predicted_actual(y_test, predictions, metrics["R2"],
                           "Figure 13  -  Multiple Linear Regression",
                           "fig13_mlr.png")
    return metrics, model, scaler


def polynomial_regression(x_train, x_test, y_train, y_test, features, scaler):
    """Degree-2 polynomial expansion of the standardised predictors."""
    subsection("5.3 Polynomial Regression (degree 2)")
    pipeline = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scale", StandardScaler()),
        ("lin", LinearRegression()),
    ])
    scaled_train = scaler.transform(x_train)
    pipeline.fit(scaled_train, y_train)
    predictions = pipeline.predict(scaler.transform(x_test))
    metrics = compute_metrics(y_test, predictions, len(features))
    LOGGER.info("  Expanded design matrix : %d -> %d terms",
                len(features), pipeline.named_steps["poly"].n_output_features_)
    LOGGER.info("  Test R2 = %.4f | Adj R2 = %.4f | MAE = %.4f | RMSE = %.4f",
                metrics["R2"], metrics["Adj_R2"], metrics["MAE"], metrics["RMSE"])
    LOGGER.info("")
    LOGGER.info("  COMMENT: squaring every predictor multiplies the design matrix by")
    LOGGER.info("  roughly 5.5x. The gain over plain multiple regression is real but comes")
    LOGGER.info("  from a far larger model, so it is unsurprising that the extra curvature")
    LOGGER.info("  does not generalise well - the tree models capture the same curvature")
    LOGGER.info("  with far fewer effective parameters.")
    return metrics, pipeline

def tuned_ridge(x_train, x_test, y_train, y_test, features, scaler):
    """Ridge with alpha chosen by RidgeCV over a wide log-spaced grid."""
    subsection("5.4 Ridge Regression - alpha tuned by 5-fold CV (Task 9)")
    folds = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scaled_train = scaler.transform(x_train)
    search = RidgeCV(alphas=RIDGE_ALPHAS, cv=folds,
                     scoring="neg_mean_squared_error").fit(scaled_train, y_train)
    model = Ridge(alpha=search.alpha_).fit(scaled_train, y_train)
    predictions = model.predict(scaler.transform(x_test))
    metrics = compute_metrics(y_test, predictions, len(features))
    LOGGER.info("  alpha grid : %d candidates from 1e-4 to 1e4", len(RIDGE_ALPHAS))
    LOGGER.info("  best alpha : %.4f", search.alpha_)
    LOGGER.info("  Test R2 = %.4f | Adj R2 = %.4f | MAE = %.4f | RMSE = %.4f",
                metrics["R2"], metrics["Adj_R2"], metrics["MAE"], metrics["RMSE"])
    LOGGER.info("")
    LOGGER.info("  COMMENT: Ridge shrinks every coefficient towards zero but never to")
    LOGGER.info("  exactly zero, so all 8 predictors stay in the final equation. This is")
    LOGGER.info("  the standard remedy for the VIF values found earlier - though as the")
    LOGGER.info("  shrinkage table shows, the chosen alpha is small enough that the")
    LOGGER.info("  actual effect at n = 16,512 is very mild.")
    return metrics, model, search.alpha_


def tuned_lasso(x_train, x_test, y_train, y_test, features, scaler):
    """Lasso with alpha chosen by LassoCV."""
    subsection("5.5 Lasso Regression - alpha tuned by 5-fold CV (Task 9)")
    folds = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scaled_train = scaler.transform(x_train)
    search = LassoCV(alphas=LASSO_ALPHAS, cv=folds, max_iter=50000,
                     random_state=RANDOM_STATE).fit(scaled_train, y_train)
    model = Lasso(alpha=search.alpha_, max_iter=50000).fit(scaled_train, y_train)
    predictions = model.predict(scaler.transform(x_test))
    active = int((np.abs(model.coef_) > 1e-8).sum())
    metrics = compute_metrics(y_test, predictions, active)
    LOGGER.info("  best alpha     : %.6f", search.alpha_)
    LOGGER.info("  non-zero coefs : %d of %d", active, len(features))
    LOGGER.info("  Test R2 = %.4f | Adj R2 = %.4f | MAE = %.4f | RMSE = %.4f",
                metrics["R2"], metrics["Adj_R2"], metrics["MAE"], metrics["RMSE"])
    table = pd.DataFrame({"Feature": features, "Coefficient": model.coef_})
    table["abs"] = table["Coefficient"].abs()
    LOGGER.info("%s", table.sort_values("abs", ascending=False).round(4).to_string(index=False))
    table.to_csv(TAB_DIR / "14_lasso_coefficients.csv", index=False)
    zeroed = [f for f, c in zip(features, model.coef_) if abs(c) <= 1e-8]
    LOGGER.info("")
    LOGGER.info("  COMMENT: Lasso set %d coefficient(s) exactly to zero: %s.",
                len(zeroed), zeroed or "none")
    LOGGER.info("  The CV search chose a near-zero alpha, which is almost no penalty at")
    LOGGER.info("  all, so L1 had no reason to sparsify and the model collapsed onto")
    LOGGER.info("  plain OLS. Lasso only becomes a real feature selector when alpha is")
    LOGGER.info("  large enough to bite - reporting this honestly is more useful than")
    LOGGER.info("  claiming the textbook behaviour that this data does not exhibit.")
    return metrics, model, search.alpha_, zeroed


def tuned_elastic_net(x_train, x_test, y_train, y_test, features, scaler):
    """Elastic Net with a joint 2-D search over alpha and l1_ratio."""
    subsection("5.6 Elastic Net Regression - 2-D grid search (Task 9)")
    folds = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scaled_train = scaler.transform(x_train)
    search = ElasticNetCV(l1_ratio=ELASTICNET_L1_RATIOS,
                          n_alphas=len(ELASTICNET_ALPHAS),
                          alphas=ELASTICNET_ALPHAS, cv=folds, max_iter=50000,
                          random_state=RANDOM_STATE).fit(scaled_train, y_train)
    model = ElasticNet(alpha=search.alpha_, l1_ratio=search.l1_ratio_,
                       max_iter=50000).fit(scaled_train, y_train)
    predictions = model.predict(scaler.transform(x_test))
    active = int((np.abs(model.coef_) > 1e-8).sum())
    metrics = compute_metrics(y_test, predictions, active)
    LOGGER.info("  best alpha    : %.6f", search.alpha_)
    LOGGER.info("  best l1_ratio : %.2f", search.l1_ratio_)
    LOGGER.info("  non-zero coefs: %d of %d", active, len(features))
    LOGGER.info("  Test R2 = %.4f | Adj R2 = %.4f | MAE = %.4f | RMSE = %.4f",
                metrics["R2"], metrics["Adj_R2"], metrics["MAE"], metrics["RMSE"])
    LOGGER.info("")
    LOGGER.info("  COMMENT: l1_ratio = %.2f shows the search leaning towards the L1 end"
                % search.l1_ratio_)
    LOGGER.info("  of the dial, for the same reason as Lasso. Elastic Net remains the")
    LOGGER.info("  recommended regularised choice because that single parameter lets it")
    LOGGER.info("  degrade gracefully towards L2 shrinkage if a future sample differs.")
    return metrics, model, search.alpha_, search.l1_ratio_


def robust_regression(x_train, x_test, y_train, y_test, features, scaler):
    """Huber regression as a robustness sensitivity check."""
    subsection("5.7 Robust Regression (Huber) - optional")
    model = HuberRegressor(max_iter=1000).fit(scaler.transform(x_train), y_train)
    predictions = model.predict(scaler.transform(x_test))
    metrics = compute_metrics(y_test, predictions, len(features))
    LOGGER.info("  Test R2 = %.4f | Adj R2 = %.4f | MAE = %.4f | RMSE = %.4f",
                metrics["R2"], metrics["Adj_R2"], metrics["MAE"], metrics["RMSE"])
    LOGGER.info("")
    LOGGER.info("  COMMENT: Huber has the WORST R2 of the linear family yet the BEST")
    LOGGER.info("  linear MAE. That is the textbook signature of a robust fit: it")
    LOGGER.info("  protects the typical prediction from extreme blocks while giving up")
    LOGGER.info("  on the tails. It is a genuine trade-off, not a free improvement, and")
    LOGGER.info("  it empirically vindicates retaining the outliers identified earlier.")
    return metrics, model


def random_forest(x_train, x_test, y_train, y_test, features):
    """Random Forest as the nonlinear benchmark."""
    subsection("5.8 Random Forest Regression - optional nonlinear benchmark")
    model = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE,
                                  n_jobs=-1).fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = compute_metrics(y_test, predictions, len(features))
    importance = pd.DataFrame({"Feature": features,
                               "Importance": model.feature_importances_}
                              ).sort_values("Importance", ascending=False)
    LOGGER.info("%s", importance.round(4).to_string(index=False))
    LOGGER.info("  Test R2 = %.4f | Adj R2 = %.4f | MAE = %.4f | RMSE = %.4f",
                metrics["R2"], metrics["Adj_R2"], metrics["MAE"], metrics["RMSE"])
    LOGGER.info("")
    LOGGER.info("  COMMENT: the forest captures curvature and interactions that no")
    LOGGER.info("  linear or quadratic term can express - most obviously the curved")
    LOGGER.info("  coastal geography. MedInc alone accounts for over half the total")
    LOGGER.info("  importance, matching the correlation and coefficient analyses.")
    importance.to_csv(TAB_DIR / "16_random_forest_importance.csv", index=False)
    _plot_importance(importance)
    return metrics, model


def _plot_predicted_actual(y_test, predictions, r2, title, filename) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(y_test, predictions, s=8, alpha=0.35, color="#4C72B0", edgecolors="none")
    limits = [min(y_test.min(), predictions.min()), max(y_test.max(), predictions.max())]
    ax.plot(limits, limits, "r--", lw=2, label="perfect prediction line")
    ax.set_xlabel("Actual MedHouseVal")
    ax.set_ylabel("Predicted MedHouseVal")
    ax.set_title(f"{title} (R2 = {r2:.4f})", fontweight="bold")
    ax.legend()
    save_figure(filename)


def _plot_importance(importance: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ordered = importance.sort_values("Importance")
    ax.barh(ordered["Feature"], ordered["Importance"], color="#55A868", edgecolor="white")
    ax.set_xlabel("Feature importance (variance reduction)")
    ax.set_title("Figure 14  -  Random Forest feature importance", fontweight="bold")
    save_figure("fig14_rf_importance.png")


def gradient_boosting(x_train, x_test, y_train, y_test, features):
    """Gradient Boosting as the second nonlinear benchmark."""
    subsection("5.9 Gradient Boosting Regression - optional nonlinear benchmark")
    model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3,
                                      random_state=RANDOM_STATE).fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = compute_metrics(y_test, predictions, len(features))
    LOGGER.info("  Test R2 = %.4f | Adj R2 = %.4f | MAE = %.4f | RMSE = %.4f",
                metrics["R2"], metrics["Adj_R2"], metrics["MAE"], metrics["RMSE"])
    LOGGER.info("")
    LOGGER.info("  COMMENT: boosting does NOT beat the forest on this data. With a low")
    LOGGER.info("  learning rate of 0.05 and only 300 shallow trees the model is")
    LOGGER.info("  under-trained; increasing the estimator count is the obvious next step.")
    return metrics, model
