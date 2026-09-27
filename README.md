# Regression Analysis - California Housing

A reproducible implementation of a **regression analysis lab assignment** covering ten
mandatory tasks on the **California Housing** dataset (Dataset 1 of 16).

> **20,640 observations | 8 predictors | target: median house value (USD 100,000)**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: PEP8](https://img.shields.io/badge/code%20style-PEP8-brightgreen.svg)](https://peps.python.org/pep-0008/)

---

## Problem statement

Predict the median house value of a California census block from eight socio-demographic
and geographic variables, then determine which candidate model is fit for purpose,
**documenting and justifying every decision with statistical evidence**.

This dataset is a good test of diagnostic ability because it is completely clean yet
**deliberately violates three of the five classical regression assumptions**. The
interesting work is diagnostic, not mechanical.

---

## Tasks covered

| # | Mandatory task | Where |
|---|---|---|
| 1 | Exploratory Data Analysis | [`eda.py`](src/regression_analysis/eda.py) |
| 2 | Correlation analysis | [`diagnostics.py`](src/regression_analysis/diagnostics.py) |
| 3 | Outlier identification | [`outliers.py`](src/regression_analysis/outliers.py) |
| 4 | Multicollinearity analysis using VIF | [`vif.py`](src/regression_analysis/vif.py) |
| 5 | Simple / Multiple / Polynomial / Ridge / Lasso / Elastic Net | [`models.py`](src/regression_analysis/models.py) |
| 5b | *Optional:* Robust (Huber) + Random Forest / Gradient Boosting | [`models.py`](src/regression_analysis/models.py) |
| 6 | Model comparison: R2, Adj R2, MAE, MSE, RMSE + residual analysis | [`comparison.py`](src/regression_analysis/comparison.py) |
| 7 | Assumption checks: linearity, independence, homoscedasticity, normality, multicollinearity | [`assumptions.py`](src/regression_analysis/assumptions.py) |
| 8 | Feature significance and coefficient interpretation | [`coefficients.py`](src/regression_analysis/coefficients.py) |
| 9 | Cross-validated hyperparameter selection for Ridge / Lasso / Elastic Net | [`models.py`](src/regression_analysis/models.py) |
| 10 | Final model selection with justification | [`comparison.py`](src/regression_analysis/comparison.py) |

The full written report with every output table, comment and conclusion is in
**[`REPORT.md`](REPORT.md)**.

---

## Key results

### Model comparison (held-out test set, 4,128 rows)

| Model | R2 | Adj R2 | MAE | RMSE | MAE (USD) |
|---|---|---|---|---|---|
| **Random Forest** | **0.8062** | **0.8058** | **0.3268** | **0.5040** | **$32,681** |
| Gradient Boosting | 0.7925 | 0.7921 | 0.3550 | 0.5215 | $35,500 |
| Polynomial (deg 2) | 0.6457 | 0.6450 | 0.4670 | 0.6814 | $46,700 |
| Lasso | 0.5769 | 0.5760 | 0.5331 | 0.7446 | $53,316 |
| Elastic Net | 0.5768 | 0.5760 | 0.5331 | 0.7447 | $53,316 |
| Ridge | 0.5759 | 0.5751 | 0.5332 | 0.7455 | $53,317 |
| Multiple Linear (OLS) | 0.5758 | 0.5750 | 0.5332 | 0.7456 | $53,320 |
| Robust (Huber) | 0.5610 | 0.5602 | 0.5158 | 0.7584 | $51,585 |
| Simple Linear | 0.4589 | 0.4587 | 0.6299 | 0.8421 | $62,991 |

### Assumption tests: 2 of 5 satisfied

| Assumption | Test | Statistic | p-value | Result |
|---|---|---|---|---|
| Linearity | Overall F-test | F = 3261.38 | 0.0 | **PASS** |
| Independence | Durbin-Watson | 1.9618 | - | **PASS** |
| Homoscedasticity | Breusch-Pagan | LM = 1170.51 | 2.25e-247 | **FAIL** |
| Normality | Jarque-Bera | 9371.47 | 0.0 | **FAIL** |
| Multicollinearity | max VIF | 9.2061 | - | **FAIL** |


### Three findings worth remembering

1. **AveBedrms correlates -0.047 with the target (essentially zero) yet carries the fourth
   largest coefficient (+$33,926).** It has no *marginal* relationship but a strong
   *partial* one. Correlation-based feature selection would discard a critical predictor.

2. **AveRooms has a negative coefficient while AveBedrms is positive, and both are
   significant.** This is a textbook suppression effect from their 0.85 correlation.
   Read alone, you would wrongly conclude that more rooms reduce house value.

3. **The target is hard-censored at $500,005.** This single design fact explains the
   heteroscedasticity, the fat residual tails, and why linear R2 plateaus near 0.6.

### Recommended model

| Role | Model | Why |
|---|---|---|
| **Primary (interpretable)** | **Elastic Net** (Ridge equivalent) | The brief requires coefficient interpretation and significance testing, which tree ensembles cannot provide. |
| **Benchmark (accuracy)** | **Random Forest** | Best on all five metrics; the accuracy ceiling if prediction is the only goal. |

> **Honest note on regularisation:** cross-validation selected near-zero penalties
> (Lasso alpha = 0.001, selecting nothing), so the penalised models are effectively OLS on
> this dataset. Regularisation remains the right *safeguard* given VIFs of 7-9, but the
> large sample, not the penalty, is what delivers the accuracy.

---

## Contents

| File | Purpose |
|---|---|
| `run_analysis.py` | Entry point - runs every task and writes all artefacts |
| `REPORT.md` | Full written report: outputs, comments, justifications |
| `docs/REFERENCES.md` | Dataset citation, sources per diagnostic, further reading |
| `docs/assignment/` | The original assignment brief (yellow highlights mark mandatory tasks) |
| `src/regression_analysis/` | The analysis package, one module per task group |
| `outputs/figures/` | 23 diagnostic and comparison plots |
| `outputs/tables/` | 25 result tables (CSV) |
| `outputs/console_output.txt` | Complete run log |
| `.github/workflows/ci.yml` | CI that re-runs the analysis and checks the headline result is stable |

---

## Project structure

```
.
|-- run_analysis.py              # entry point: python run_analysis.py
|-- requirements.txt
|-- REPORT.md                    # full written report
|-- LICENSE
|-- docs/
|   |-- REFERENCES.md            # citations and further reading
|   `-- assignment/              # original brief
|-- src/regression_analysis/
|   |-- __init__.py
|   |-- config.py                # paths, seed, grids (portable, no hard-coded drives)
|   |-- logging_utils.py
|   |-- utils.py                 # metrics, figure I/O, formatting
|   |-- data.py                  # loading and train/test split
|   |-- eda.py                   # Task 1
|   |-- diagnostics.py           # Task 2
|   |-- outliers.py              # Task 3
|   |-- vif.py                   # Task 4
|   |-- models.py                # Tasks 5 and 9
|   |-- comparison.py            # Tasks 6 and 10
|   |-- assumptions.py           # Task 7
|   |-- coefficients.py          # Task 8
|   `-- pipeline.py              # orchestration
`-- outputs/
    |-- figures/                 # 23 diagnostic and comparison plots
    |-- tables/                  # 25 result tables (CSV)
    `-- console_output.txt       # complete run log
```

---

## Usage

```bash
git clone https://github.com/Sanskar1724/-Regression-modelling-analysis.git
cd -Regression-modelling-analysis

python -m venv .venv
# Windows:     .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
python run_analysis.py
```

The first run downloads and caches the dataset (~1.5 MB). Expect roughly **60 seconds**
of compute, producing 23 figures and 25 tables in `outputs/`.


---

## Methodological notes

A few decisions are judgement calls rather than defaults, and are worth flagging:

- **Outliers are retained.** Three detectors were used (IQR, Mahalanobis, Cook's distance).
  The Huber sensitivity test *empirically confirms* this: down-weighting the same rows
  **lowers** R2, which would be the opposite of the case if they were corrupt data.
- **Variables are not dropped despite VIF > 5.** Two would have to go, but removing the
  coordinates would delete the strongest spatial signal in the data. Regularisation is
  the better remedy.
- **Studentised residuals use a closed-form vectorised identity.** The statsmodels estimator
  loops in Python and takes minutes at n = 16,512; the replacement agrees to ~1e-14.
- **Cross-validation is used for ranking, the held-out test set for the generalisation
  estimate.** CV R2 is consistently higher because each fold trains on 4/5 of the data.
- **Known limitation:** the random split lets neighbouring geographic blocks appear in
  both train and test, which slightly *inflates* all scores. Spatial k-fold CV would give
  a more honest estimate for geographic data.
- **Not implemented:** censored (Tobit) regression, which would be the methodologically
  correct model for the $500,005 ceiling if uncensored prices were available.

---

## References

See [`docs/REFERENCES.md`](docs/REFERENCES.md) for the dataset citation, the textbook
sources behind each diagnostic, and further reading.

---

## License

Released under the [MIT License](LICENSE). The California Housing dataset is distributed
by scikit-learn and originates from the 1990 U.S. Census.

## Author

**Sanskar** - <https://github.com/Sanskar1724>

Results are **deterministic**: `RANDOM_STATE = 42` is used for the train/test split, every
estimator and every cross-validation fold.
