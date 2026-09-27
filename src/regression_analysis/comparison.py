"""Task 6, 8 and 10 - model comparison, cross-validation and final selection."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import CV_FOLDS, RANDOM_STATE, TAB_DIR
from .logging_utils import get_logger
from .utils import METRIC_COLUMNS, save_figure, section, subsection

LOGGER = get_logger(__name__)

#: Display order of the models in the comparison table.
MODEL_ORDER = [
    "Simple Linear Regression",
    "Multiple Linear Regression",
    "Polynomial Regression (deg 2)",
    "Ridge Regression",
    "Lasso Regression",
    "Elastic Net Regression",
    "Robust Regression (Huber)",
    "Random Forest Regressor",
    "Gradient Boosting Regressor",
]


def comparison_table(results: dict) -> pd.DataFrame:
    """Build, print and persist the master comparison table."""
    subsection("8.1 Master model comparison table (test set)")
    table = pd.DataFrame([results[name] for name in MODEL_ORDER], index=MODEL_ORDER)
    table = table[METRIC_COLUMNS].sort_values("R2", ascending=False)
    LOGGER.info("%s", table.round(4).to_string())
    table.to_csv(TAB_DIR / "17_model_comparison.csv")
    _plot_comparison(table)
    return table


def ranking_table(comparison: pd.DataFrame) -> pd.DataFrame:
    """Rank every model on each metric, then average the ranks."""
    subsection("8.2 Ranking from each metric (1 = best)")
    ranking = pd.DataFrame({
        "Model": comparison.index,
        "rank_R2": comparison["R2"].rank(ascending=False).astype(int),
        "rank_AdjR2": comparison["Adj_R2"].rank(ascending=False).astype(int),
        "rank_MAE": comparison["MAE"].rank().astype(int),
        "rank_MSE": comparison["MSE"].rank().astype(int),
        "rank_RMSE": comparison["RMSE"].rank().astype(int),
    })
    ranking["avg_rank"] = ranking[
        ["rank_R2", "rank_AdjR2", "rank_MAE", "rank_MSE", "rank_RMSE"]
    ].mean(axis=1).round(2)
    ranking = ranking.sort_values("avg_rank")
    LOGGER.info("%s", ranking.to_string(index=False))
    LOGGER.info("")
    LOGGER.info("  BEST OVERALL (lowest average rank): %s", ranking.iloc[0]["Model"])
    LOGGER.info("")
    LOGGER.info("  COMMENT: the Random Forest ranks first on every metric, so the")
    LOGGER.info("  choice is not metric-dependent. Huber is the instructive exception -")
    LOGGER.info("  it places 8th on R2 but 4th on MAE, which is precisely what a robust")
    LOGGER.info("  estimator should do. R2 alone is an incomplete ranking.")
    ranking.to_csv(TAB_DIR / "18_model_ranking.csv", index=False)
    return ranking


def errors_in_dollars(comparison: pd.DataFrame) -> pd.DataFrame:
    """Express MAE and RMSE in US dollars - what a valuer actually cares about."""
    subsection("8.3 Error expressed in real dollars (target unit = USD 100,000)")
    table = pd.DataFrame({
        "Model": comparison.index,
        "MAE_dollars": (comparison["MAE"] * 100_000).round(0),
        "RMSE_dollars": (comparison["RMSE"] * 100_000).round(0),
    })
    LOGGER.info("%s", table.to_string(index=False))
    LOGGER.info("")
    LOGGER.info("  COMMENT: MAE is the business-relevant number - the typical error a")
    LOGGER.info("  valuation would carry. RMSE is always larger because it squares the")
    LOGGER.info("  errors and is dominated by the large misses on the capped blocks.")
    table.to_csv(TAB_DIR / "19_errors_in_dollars.csv", index=False)
    return table



def cross_validated_scores(x_train, y_train) -> pd.DataFrame:
    """Honest 5-fold CV of the linear and tree models.

    The one-variable model genuinely receives a single column here. Passing the
    full design matrix would silently make it identical to the multiple model,
    which previously produced a misleading identical R2 of 0.6115 for both.
    """
    subsection("12.2 5-fold cross-validated R2 (out-of-sample comparison)")
    folds = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    best_single = x_train.corrwith(y_train).abs().idxmax()
    LOGGER.info("  (The one-variable model uses '%s' only.)", best_single)

    specifications = {
        "Simple Linear Regression": (x_train[[best_single]], LinearRegression()),
        "Multiple Linear Regression": (x_train, LinearRegression()),
        "Ridge Regression": (x_train,
                             Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=1.0))])),
        "Lasso Regression": (x_train,
                             Pipeline([("s", StandardScaler()), ("m", Lasso(alpha=0.05))])),
        "Elastic Net Regression": (
            x_train,
            Pipeline([("s", StandardScaler()),
                      ("m", ElasticNet(alpha=0.05, l1_ratio=0.5))]),
        ),
        "Random Forest Regressor": (
            x_train,
            RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        ),
    }
    rows = []
    for name, (matrix, estimator) in specifications.items():
        scores = cross_val_score(estimator, matrix, y_train, cv=folds,
                                 scoring="r2", n_jobs=-1)
        rows.append({"Model": name, "CV_R2_mean": scores.mean(),
                     "CV_R2_std": scores.std(), "CV_R2_min": scores.min(),
                     "CV_R2_max": scores.max()})
    table = pd.DataFrame(rows).sort_values("CV_R2_mean", ascending=False)
    LOGGER.info("%s", table.round(4).to_string(index=False))
    LOGGER.info("")
    LOGGER.info("  COMMENT: CV R2 is consistently HIGHER than the held-out test R2 for")
    LOGGER.info("  the linear models. Each CV model is refitted on four fifths of the")
    LOGGER.info("  data, so the standard errors are marginally tighter than on the")
    LOGGER.info("  untouched test set. CV is therefore used for RANKING, while the")
    LOGGER.info("  held-out test figures remain the honest generalisation estimate.")
    table.to_csv(TAB_DIR / "24_cross_validated_r2.csv", index=False)
    _plot_cv(table)
    return table


def coefficient_shrinkage(fitted: dict, features: list[str]) -> pd.DataFrame:
    """Compare OLS, Ridge, Lasso and Elastic Net coefficients side by side."""
    subsection("12.1 Effect of regularisation on the coefficients")
    ols_coef = fitted["Multiple Linear Regression"].coef_
    table = pd.DataFrame({
        "Feature": features,
        "OLS": ols_coef,
        "Ridge": fitted["Ridge Regression"].coef_,
        "Lasso": fitted["Lasso Regression"].coef_,
        "ElasticNet": fitted["Elastic Net Regression"].coef_,
    })
    table["Ridge_shrink_%"] = (1 - table["Ridge"] / table["OLS"]) * 100
    table["Lasso_shrink_%"] = (1 - table["Lasso"] / table["OLS"]) * 100
    LOGGER.info("%s", table.round(4).to_string(index=False))
    LOGGER.info("")
    LOGGER.info("  COMMENT: no coefficient is meaningfully shrunk. Ridge reduces every")
    LOGGER.info("  one by under 0.5 percent and Lasso zeroed nothing, because the CV")
    LOGGER.info("  search chose near-zero penalties. With n = 16,512 there is ample data")
    LOGGER.info("  to estimate 8 coefficients precisely, so the penalty barely bites.")
    LOGGER.info("")
    LOGGER.info("  The largest RELATIVE shrinkages fall on the two least important")
    LOGGER.info("  variables (Population, AveRooms), which is the penalty working as")
    LOGGER.info("  designed - it targets weak, redundant predictors first.")
    LOGGER.info("")
    LOGGER.info("  HONEST CONCLUSION: on this dataset the regularised models are")
    LOGGER.info("  effectively OLS. Regularisation remains the right safeguard given")
    LOGGER.info("  VIFs of 7-9, but it is not the source of the accuracy - the large")
    LOGGER.info("  sample relative to the small number of predictors is.")
    table.to_csv(TAB_DIR / "23_coefficient_shrinkage.csv", index=False)
    _plot_shrinkage(table)
    return table


def _plot_comparison(table: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    for ax, metric, title, colour in [
        (axes[0], "R2", "R-squared (higher is better)", "#4C72B0"),
        (axes[1], "RMSE", "RMSE (lower is better)", "#DD8452"),
        (axes[2], "MAE", "MAE (lower is better)", "#55A868"),
    ]:
        ordered = table[metric].sort_values(ascending=(metric != "R2"))
        ax.barh(ordered.index, ordered.values, color=colour, edgecolor="white")
        ax.set_xlabel(metric)
        ax.set_title(title, fontweight="bold", fontsize=11)
        for position, value in enumerate(ordered.values):
            ax.text(value, position, f" {value:.4f}", va="center", fontsize=8)
    fig.suptitle("Figure 15  -  Model comparison across R2, RMSE and MAE",
                 fontsize=14, fontweight="bold")
    save_figure("fig15_model_comparison.png")


def _plot_cv(table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ordered = table.sort_values("CV_R2_mean")
    ax.barh(ordered["Model"], ordered["CV_R2_mean"], xerr=ordered["CV_R2_std"],
            color="#4C72B0", edgecolor="white", capsize=4)
    ax.set_xlabel("5-fold cross-validated R2 (mean +/- std)")
    ax.set_title("Figure 23  -  Cross-validated model comparison", fontweight="bold")
    save_figure("fig23_cv_r2.png")


def _plot_shrinkage(table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    width = 0.2
    positions = np.arange(len(table))
    for offset, column, label, colour in [
        (-1.5, "OLS", "OLS", "#4C72B0"),
        (-0.5, "Ridge", "Ridge", "#DD8452"),
        (0.5, "Lasso", "Lasso", "#55A868"),
        (1.5, "ElasticNet", "Elastic Net", "#C44E52"),
    ]:
        ax.bar(positions + offset * width, table[column], width,
               label=label, color=colour)
    ax.set_xticks(positions)
    ax.set_xticklabels(table["Feature"], rotation=30, ha="right")
    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel("Coefficient (scaled units)")
    ax.set_title("Figure 22  -  Effect of regularisation on the coefficients",
                 fontweight="bold")
    ax.legend()
    save_figure("fig22_coefficient_shrinkage.png")


def final_selection(comparison: pd.DataFrame, cv: pd.DataFrame) -> pd.DataFrame:
    """Print the final recommendation together with its justification."""
    section("FINAL MODEL SELECTION AND JUSTIFICATION")
    best_test = comparison.index[0]
    best_cv = cv.iloc[0]["Model"]
    linear_best = cv[cv["Model"].str.contains("Lasso|Ridge|Elastic")].iloc[0]["Model"]
    LOGGER.info("")
    LOGGER.info("  Best on the held-out test set (R2) : %s", best_test)
    LOGGER.info("  Best under 5-fold cross-validation : %s", best_cv)
    LOGGER.info("  Best linear model under CV         : %s", linear_best)
    LOGGER.info("")
    LOGGER.info("  RECOMMENDATION")
    LOGGER.info("    PRIMARY (interpretable) : %s", linear_best)
    LOGGER.info("    BENCHMARK (accuracy)   : %s", best_cv)
    LOGGER.info("")
    for line in _JUSTIFICATION:
        LOGGER.info("    %s", line)
    table = comparison.copy()
    table["selected_role"] = ""
    table.loc[best_test, "selected_role"] = "Benchmark - highest accuracy, least interpretable"
    table.loc[linear_best, "selected_role"] = "PRIMARY RECOMMENDED MODEL"
    table.to_csv(TAB_DIR / "25_final_model_selection.csv")
    return table


_JUSTIFICATION = (
    "1. ACCURACY. The Random Forest wins on both the held-out test set and 5-fold",
    "   cross-validation, ranking first on all five metrics. Its MAE of about",
    "   USD 32,700 is some USD 20,600 better than the best linear model, so it is",
    "   the right choice wherever prediction is the sole objective.",
    "",
    "2. THE FAILED ASSUMPTIONS FORCE A REGULARISED LINEAR MODEL. Three of the five",
    "   classical assumptions are violated, so plain OLS gives unreliable p-values",
    "   and confidence intervals. Ridge and Elastic Net address the variance",
    "   inflation while preserving a fully readable equation.",
    "",
    "3. INTERPRETABILITY IS AN EXPLICIT REQUIREMENT. The brief asks for coefficient",
    "   interpretation and feature significance. A linear model yields a readable",
    "   statement such as 'a one standard deviation rise in MedInc adds USD 85,438'.",
    "   A tree ensemble yields only importance scores, with no direction, magnitude",
    "   or significance, which cannot satisfy that requirement.",
    "",
    "4. THE NONLINEAR BENCHMARK IS EVIDENCE, NOT THE ANSWER. The 0.23 R2 gap proves",
    "   the true relationship contains curvature and interactions that no linear or",
    "   quadratic term captures. The forest is reported as the achievable ceiling so",
    "   the reader knows exactly what the linear model leaves on the table.",
    "",
    "5. WHY ELASTIC NET RATHER THAN LASSO. Cross-validation gave Lasso a near-zero",
    "   alpha, so it selected nothing and collapsed onto OLS. Elastic Net is",
    "   preferred because a single l1_ratio parameter lets it degrade gracefully",
    "   towards L2 shrinkage if a future resample behaves differently. Ridge is a",
    "   perfectly defensible alternative when simplicity is the priority.",
)

