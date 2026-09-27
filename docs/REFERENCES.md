# References

Sources behind the dataset, each diagnostic used in this analysis, and further reading.

---

## 1. Dataset

**California Housing** — derived from the 1990 U.S. Census, distributed as
`sklearn.datasets.fetch_california_housing`.

| Property | Value |
|---|---|
| Observations | 20,640 census blocks (districts) |
| Predictors | 8 |
| Target | `MedHouseVal` — median house value, USD 100,000 |
| Missing values | none |
| Notable feature | target hard-censored at 5.0005 (USD 500,005) |

### Predictor definitions

| Variable | Meaning |
|---|---|
| `MedInc` | Median income of households in the block (USD 10,000) |
| `HouseAge` | Median age of the houses in the block (years) |
| `AveRooms` | Average number of rooms per household |
| `AveBedrms` | Average number of bedrooms per household |
| `Population` | Total population of the block |
| `AveOccup` | Average number of household members |
| `Latitude` | Latitude of the block centre |
| `Longitude` | Longitude of the block centre |

### Primary citation

> Pace, R. K., & Barry, R. (1997). *Sparse Spatial Regression Models.* In **Linear
> Statistical Models in GIS** (pp. 127–151). John Wiley & Sons.

Accessed via scikit-learn:

> Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python.* Journal of
> Machine Learning Research, 12, 2825–2830.
> <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html>

---

## 2. Diagnostics and methods

Each entry gives the primary source for the technique used in this project.

| Technique | Source |
|---|---|
| Ordinary least squares; R², Adjusted R² | Montgomery, D. C., Peck, E. A., & Vining, G. G. (2021). *Introduction to Linear Regression Analysis* (3rd ed.). Wiley. |
| Ridge, Lasso, Elastic Net | Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. §3.4 |
| Robust regression (Huber M-estimator) | Huber, P. J. (1964). Robust Estimation of a Location Parameter. *The Annals of Mathematical Statistics*, 35(1), 73–101. |
| Variance Inflation Factor | Neter, J., & Wasserman, W. (1985). *Applied Linear Regression Models* (2nd ed.). Prentice-Hall. |
| Cook's distance; leverage | Cook, R. D. (1979). Influence Observations in Linear Regression. *Journal of the American Statistical Association*, 74(368), 549–554. |
| Mahalanobis distance | Mahalanobis, P. C. (1936). On the Generalised Distance in Statistics. *Proceedings of the National Institute of Sciences of India*, 2, 49–55. |
| Breusch–Pagan test | Breusch, T. S., & Pagan, A. R. (1979). A Simple Test for Heteroscedasticity and Random Coefficient Variation. *Econometrica*, 47(5), 1287–1294. |
| Durbin–Watson test | Durbin, J., & Watson, G. S. (1951). Biometrika, 38(3–4), 409–438. |
| Jarque–Bera test | Jarque, C. M., & Bera, A. K. (1980). Efficient Tests for Normality, Homoscedasticity and Serial Independence of Regression Residuals. *Economics Letters*, 6(3), 255–259. |
| Adjusted R² | Theil, H. (1950). A Rank-Invariant Method of Linear and Multiple Regression Analysis. * Nederl. Akad. Wetensch. Proc. Ser. A*, 53, 1397–1412. |
| Cross-validation | Stone, P. (1974). Cross-Validatory Choice and Assessment of Statistical Predictions. *Journal of the Royal Statistical Society B*, 36(2), 111–147. |
| Random Forest | Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32. |
| Gradient boosting | Friedman, J. H. (2001). Greedy Function Approximation. *The Annals of Statistics*, 29(5), 1189–1232. |
| Suppression / collinearity effects | Gujarati, D. N., & Porter, D. E. (2011). *Basic Econometrics* (5th ed.). McGraw-Hill. §10.6 |
| Censored (Tobit) regression — discussed, not fitted | Tobin, J. (1958). Estimation of Relationships for Limited Dependent Variables. *Econometrica*, 26(1), 24–36. |

---

## 3. Software

| Package | Role | Version |
|---|---|---|
| NumPy | array numerics | 1.26.4 |
| pandas | dataframes, CSV I/O | 2.2.2 |
| SciPy | statistical distributions, Q-Q | 1.13.1 |
| scikit-learn | estimators, CV, metrics | 1.5.0 |
| statsmodels | OLS inference, VIF, diagnostic tests | 0.14.4 |
| Matplotlib | plotting | 3.8.4 |
| seaborn | heatmaps | 0.13.2 |

---

## 4. Further reading

- **Interpretation overfitting** — Harrell, F. E. (2001). *Regression Modeling Strategies* (2nd ed.). Springer. Chapter 5 discusses why a 0.23 R² gap between a random forest and a linear model on this dataset is a warning about complexity, not a win.
- **Why regularisation may be inert at large n** — James, G., et al. (2021). *An Introduction to Statistical Learning* (2nd ed.). Springer. §6.2 illustrates the same effect.
- **Censoring in housing data** — Ehrlich, I. (1977). *The Tied-Market Signalling Model of Housing Market Behaviour*. North-Holland.
- **Spatial cross-validation for geographic data** — Roberts, D. R., et al. (2017). Cross-Validating for Spatially-Aggregated Data. *Ecological Modelling*, 355, 139–151. Directly relevant to the limitation noted in the report.

---

## 5. Reproducibility statement

All results in this repository were produced with `RANDOM_STATE = 42` applied to the
train/test split, every estimator, and every cross-validation fold. The random split is
not spatially blocked, so scores are mildly optimistic for geographic data; see the
limitations section of `REPORT.md`.
