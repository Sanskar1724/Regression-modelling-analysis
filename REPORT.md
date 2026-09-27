# Regression Analysis Report - California Housing

**Dataset selected:** Dataset 1 - **California Housing**
**Domain:** Real Estate / Economics | **Size:** 20,640 observations x 8 predictors
**Target:** `MedHouseVal` - median house value, in units of $100,000
**Source:** `sklearn.datasets.fetch_california_housing` (derived from the 1990 California census)

**Analysis script:** `run_analysis.py` (package in `src/regression_analysis/`)
**Raw output:** `outputs/console_output.txt` | 23 figures | 25 tables
**Citations:** [`docs/REFERENCES.md`](docs/REFERENCES.md)

---

## 1. Problem statement

Build a model predicting the median house value of a California census block from eight
socio-demographic and geographic variables, then establish which candidate model is fit for
purpose, **documenting and justifying every decision** with statistical evidence.

**Why this dataset suits the assignment:** it is completely clean (no missing values), runs on
a laptop, and - importantly - it deliberately *violates* three of the five classic regression
assumptions. That makes it a real test of diagnostic ability rather than a formality.

**Method:** 80/20 train-test split (seed 42) -> EDA -> correlation -> outliers -> VIF ->
nine models -> five assumption tests -> coefficient interpretation -> cross-validated
final selection.

---

## 2. Task 1 - Exploratory Data Analysis

### Output

```
Dataset shape : (20640, 10)    Total missing cells : 0    Infinite values : 0
```

| Variable | mean | std | min | 25% | 50% | 75% | max | skew | kurtosis |
|---|---|---|---|---|---|---|---|---|---|
| MedInc | 3.871 | 1.900 | 0.500 | 2.563 | 3.535 | 4.743 | 15.000 | 1.647 | 4.953 |
| HouseAge | 28.639 | 12.586 | 1.000 | 18.000 | 29.000 | 37.000 | 52.000 | 0.060 | -0.801 |
| AveRooms | 5.429 | 2.474 | 0.846 | 4.441 | 5.229 | 6.052 | 141.909 | 20.698 | 879.353 |
| AveBedrms | 1.097 | 0.474 | 0.333 | 1.006 | 1.049 | 1.100 | 34.067 | 31.317 | 1636.712 |
| Population | 1425.5 | 1132.5 | 3.000 | 787.0 | 1166.0 | 1725.0 | 35682.0 | 4.936 | 73.553 |
| AveOccup | 3.071 | 10.386 | 0.692 | 2.430 | 2.818 | 3.282 | 1243.333 | 97.640 | 10651.011 |
| Latitude | 35.632 | 2.136 | 32.540 | 33.930 | 34.260 | 37.710 | 41.950 | 0.466 | -1.118 |
| Longitude | -119.570 | 2.004 | -124.350 | -121.800 | -118.490 | -118.010 | -114.310 | -0.298 | -1.330 |
| **MedHouseVal** | **2.069** | **1.154** | **0.150** | **1.196** | **1.797** | **2.647** | **5.000** | **0.978** | **0.328** |

Target bands: Low (<1.5) 7,620 (36.9%) | Moderate 9,184 (44.5%) | High 2,092 (10.1%) |
Very high 1,744 (8.5%).

*Figures 1-5 | Tables 1-4*

### Comment and justification

1. **No missing or infinite values.** No imputation or deletion is needed, so preprocessing
   introduces no risk of bias. Unusual, and worth stating explicitly.

2. **The most important fact: the target is censored at 5.0005 ($500,005).** This hard ceiling
   in the source data means no model can predict above it, so predictions for the most expensive
   houses are *systematically biased downwards*. This single design feature explains the
   heteroscedasticity in Task 7, the fat residual tails, and why R2 plateaus near 0.6 for
   linear models. A censored (Tobit) model would be methodologically correct if the uncensored
   prices were available; it is recorded as a limitation rather than fitted, since the brief
   specifies standard regression types.

3. **Severe right-skew in AveRooms, AveBedrms, AveOccup** (skew 20.7, 31.3, 97.6). These are
   per-block *ratios*: one 1,000-person household in a 2-room block yields an enormous ratio.
   This justifies the log-transform option in the brief and explains the Q-Q departure in Task 7.

4. **Latitude/Longitude are not independent axes** but a curved coastline (Figure 4). This

---

## 3. Task 2 - Correlation Analysis

### Output

| Predictor | Pearson r | R2 | Strength |
|---|---|---|---|
| **MedInc** | **+0.6881** | **0.4734** | moderate |
| AveRooms | +0.1519 | 0.0231 | weak |
| Latitude | -0.1442 | 0.0208 | weak |
| HouseAge | +0.1056 | 0.0112 | weak |
| AveBedrms | -0.0467 | 0.0022 | weak |
| Longitude | -0.0460 | 0.0021 | weak |
| Population | -0.0246 | 0.0006 | weak |
| AveOccup | -0.0237 | 0.0006 | weak |

Predictor pairs exceeding |r| = 0.7: **Latitude <-> Longitude -0.9247** (HIGH) |
**AveRooms <-> AveBedrms +0.8476** (HIGH)

*Figures 6-8 | Tables 5-7*

### Comment and justification

- **MedInc alone explains 47.3% of target variance** (r2 = 0.473) - by far the strongest single
  driver - while the next five predictors each explain under 2.5%. This immediately shows a
  one-variable model is insufficient.
- **Only one predictor has a meaningful marginal relationship.** Correlation is a *marginal*
  measure, so the near-zero values for AveBedrms, Population and AveOccup must **not** be read as
  "these variables are useless". Task 8 shows AveBedrms has the **fourth largest coefficient**
  once other variables are controlled for. This contrast is the clearest demonstration in the
  whole analysis of why bivariate correlation cannot replace a multiple regression.
- **The two HIGH pairs are the first evidence of multicollinearity**, confirmed by VIF in Task 4.
  The Latitude-Longitude pair at -0.92 is *structural* (coordinates of a coastline), not a defect.

---

## 4. Task 3 - Outlier Identification

### Output

**Univariate (1.5 x IQR rule):**

| Column | Q1 | Q3 | IQR | Lower | Upper | n outliers | % |
|---|---|---|---|---|---|---|---|
| AveBedrms | 1.006 | 1.100 | 0.093 | 0.866 | 1.240 | 1,424 | 6.90 |
| Population | 787 | 1725 | 938 | -620 | 3132 | 1,196 | 5.79 |
| AveOccup | 2.430 | 3.282 | 0.853 | 1.151 | 4.561 | 711 | 3.44 |
| MedInc | 2.563 | 4.743 | 2.180 | -0.706 | 8.013 | 681 | 3.30 |
| AveRooms | 4.441 | 6.052 | 1.612 | 2.023 | 8.470 | 511 | 2.48 |
| MedHouseVal | 1.196 | 2.647 | 1.451 | -0.981 | 4.824 | 1,071 | 5.19 |
| HouseAge, Latitude, Longitude | - | - | - | - | - | **0** | 0.00 |

**Multivariate (Mahalanobis, p = 8):** chi2(8) 99th pct = 20.090 -> **508 rows (3.08%)** flagged.

**Influence** (leverage threshold 2k/n = 0.001090; Cook's D 4/n = 0.000242):
high-leverage 646 | studentised residual > 2: 882 | influential 762.
Most influential: row 19006, **Cook's D = 0.7740**.

*Figures 9-10 | Tables 8-10*

### Comment and conclusion

Three methods were used deliberately because each detects something different: the IQR rule works
per column, Mahalanobis distance detects unusual *combinations*, and leverage/Cook's D measures
*effect on the model itself*.

**Decision: outliers are RETAINED** - evidence-based, not assumed:

- The flagged records are **legitimate extremes, not errors**. A household of 5 in a 2-room block
  is unusual but real; deleting such rows biases coefficients away from valid observations.
- **The Huber sensitivity test in Task 7 confirms this empirically.** Down-weighting the same
  observations *lowered* R2 from 0.5758 to 0.5610. If these were corrupt data, robust regression
  would have improved the fit; it did the opposite.
- One genuine concern is acknowledged: row 19006 (Cook's D = 0.77) is extremely highly leveraged
  yet has a *low* value (1.375), making it a geographic-block outlier rather than a target
  outlier. Its influence is confined mainly to the Latitude/Longitude coefficients, already flagged
  as unstable by VIF. The honest position: this is a caution flag on *coefficient interpretation*,
  not on overall model accuracy.

---

## 5. Task 4 - Multicollinearity Analysis using VIF

### Output


---

## 6. Task 5 - Regression Models

All models fitted on standardised predictors (required for the penalised methods). 80/20 split.

### Output - model by model

| Model | Configuration | Test R2 | Adj R2 | MAE | RMSE |
|---|---|---|---|---|---|
| Simple Linear | `MedHouseVal = 0.4446 + 0.4193*MedInc` | 0.4589 | 0.4587 | 0.6299 | 0.8421 |
| Multiple Linear (OLS) | all 8 predictors | 0.5758 | 0.5750 | 0.5332 | 0.7456 |
| Polynomial (deg 2) | 8 -> **44 terms** | 0.6457 | 0.6450 | 0.4670 | 0.6814 |
| Ridge | alpha = 3.162 | 0.5759 | 0.5751 | 0.5332 | 0.7455 |
| Lasso | alpha = 0.001 | 0.5769 | 0.5760 | 0.5331 | 0.7446 |
| Elastic Net | alpha = 0.001, l1_ratio = 0.95 | 0.5768 | 0.5760 | 0.5331 | 0.7447 |
| Robust (Huber) | optional | 0.5610 | 0.5602 | 0.5158 | 0.7584 |
| Random Forest | 200 trees | **0.8062** | **0.8058** | **0.3268** | **0.5040** |
| Gradient Boosting | 300 trees, lr = 0.05 | 0.7925 | 0.7921 | 0.3550 | 0.5215 |

OLS coefficients (standardised): Latitude -0.8969 | Longitude -0.8698 | MedInc +0.8544 |
AveBedrms +0.3393 | AveRooms -0.2944 | HouseAge +0.1225 | AveOccup -0.0408 | Population -0.0023

Random Forest importances: MedInc 0.5259 | AveOccup 0.1381 | Latitude 0.0886 | Longitude 0.0883 |
HouseAge 0.0544 | AveRooms 0.0444 | Population 0.0307 | AveBedrms 0.0296

*Figures 12-14 | Tables 13, 14, 16*

### Comment and justification

- **Simple -> Multiple: R2 rises 0.4589 -> 0.5758.** The seven added predictors are worth ~0.117
  of R2 and Adjusted R2 stays at 0.5750, so they carry genuine information rather than fitting noise.
- **Polynomial (deg 2) jumps to 0.6457** by expanding 8 variables into 44 terms. The gain is real
  but comes from a 5.5x larger model, so it is unsurprising that it later proves brittle.
- **The three penalised models are all within 0.0011 of plain OLS.** This is an important and
  initially surprising result: cross-validation chose *tiny* penalties (Lasso/EN alpha = 0.001),
  so none of them meaningfully changed the fit. With n = 16,512 there is ample data to estimate
  8 coefficients precisely, so the penalty barely bites. **Regularisation is the right safeguard
  given the VIFs, but it is not the source of the accuracy here - the sample size is.**
- **Huber has the worst linear R2 (0.5610) yet the best linear MAE (0.5158).** That is the
  textbook signature of a robust fit: it protects the *typical* prediction from extreme houses
  while sacrificing the tails. It is a genuine trade-off, not a free improvement - and it
  empirically vindicates retaining the outliers (Task 3).
- **Boosting does not beat the forest** (0.7925 vs 0.8062). With lr = 0.05 and only 300 shallow
  trees it is under-trained; more estimators would likely close the gap.
- **The 0.23 R2 jump from the best linear model (0.6457) to Random Forest (0.8062) is the
  headline result.** It proves the true relationship contains curvature and interactions that
  no linear or quadratic term can express - most obviously the curved coastal geography.

---

## 7. Task 9 - Hyperparameter Selection for Ridge / Lasso / Elastic Net

All three tuned by **5-fold cross-validation** (shuffled, seed 42) on the training set.

### Output

| Model | Search space | Best alpha | Best l1_ratio | Non-zero coefs |
|---|---|---|---|---|
| Ridge | 33 candidates, 1e-4 to 1e4 (quarter-decade) | **3.1623** | n/a | 8 of 8 |
| Lasso | 13 candidates, 3.2e-3 to 3.2e3 | **0.0010** | n/a | 8 of 8 |
| Elastic Net | 13 alphas x 19 l1_ratios | **0.0010** | **0.95** | 8 of 8 |

*Table 15*

### Comment and justification

- **Lasso selected nothing.** Its CV-chosen alpha of 0.001 is essentially no penalty, so L1 had no
  reason to sparsify and the model collapsed onto plain OLS (R2 0.5769 vs 0.5758). L1 only becomes
  a genuine feature selector when alpha is large enough to bite. Reporting this honestly is more
  valuable than claiming the textbook "Lasso zeroes coefficients" behaviour.
- **Elastic Net's l1_ratio = 0.95** shows the search leaning heavily towards the L1 end of the
  dial, for the same reason.
- **Ridge is the only method that applied real shrinkage**, and even then the total coefficient
  reduction is under 0.5% for every feature - far milder than VIFs of 7-9 would suggest.
- **Honest conclusion:** on this dataset the regularised models are *effectively OLS*. This is a
  legitimate negative finding, not a failure - it follows directly from the large sample relative
  to the small number of predictors.


---

## 8. Task 6 - Model Comparison and Residual Analysis

### Output - master comparison table (test set, 4,128 rows)

| Model | R2 | Adj R2 | MAE | MSE | RMSE | Avg rank |
|---|---|---|---|---|---|---|
| **Random Forest** | **0.8062** | **0.8058** | **0.3268** | **0.2540** | **0.5040** | **1.0** |
| Gradient Boosting | 0.7925 | 0.7921 | 0.3550 | 0.2719 | 0.5215 | 2.0 |
| Polynomial (deg 2) | 0.6457 | 0.6450 | 0.4670 | 0.4643 | 0.6814 | 3.0 |
| Lasso | 0.5769 | 0.5760 | 0.5331 | 0.5545 | 0.7446 | 4.2 |
| Elastic Net | 0.5768 | 0.5760 | 0.5331 | 0.5545 | 0.7447 | 5.2 |
| Ridge | 0.5759 | 0.5751 | 0.5332 | 0.5558 | 0.7455 | 6.2 |
| Multiple Linear | 0.5758 | 0.5750 | 0.5332 | 0.5559 | 0.7456 | 7.2 |
| Robust (Huber) | 0.5610 | 0.5602 | **0.5158** | 0.5752 | 0.7584 | 7.2 |
| Simple Linear | 0.4589 | 0.4587 | 0.6299 | 0.7091 | 0.8421 | 9.0 |

**Errors in real dollars** (target unit = $100,000):

| Model | MAE ($) | RMSE ($) |
|---|---|---|
| Random Forest | **32,681** | **50,396** |
| Gradient Boosting | 35,500 | 52,148 |
| Robust (Huber) | 51,585 | 75,844 |
| Multiple Linear | 53,320 | 74,558 |
| Simple Linear | 62,991 | 84,209 |

**5-fold cross-validated R2** (fairer than a single split):

| Model | CV R2 mean | std | min | max |
|---|---|---|---|---|
| Random Forest | **0.8045** | 0.0061 | 0.7955 | 0.8127 |
| Ridge | 0.6115 | 0.0124 | 0.6008 | 0.6354 |
| Multiple Linear | 0.6115 | 0.0124 | 0.6008 | 0.6354 |
| Elastic Net | 0.5741 | 0.0111 | 0.5620 | 0.5952 |
| Lasso | 0.5470 | 0.0118 | 0.5352 | 0.5696 |
| Simple Linear | 0.4769 | 0.0116 | 0.4659 | 0.4995 |


---

## 9. Task 7 - Regression Assumption Checks

All five tests run on the OLS multiple regression fitted to the training set.

### Output - summary

| # | Assumption | Test | Statistic | p-value | Result |
|---|---|---|---|---|---|
| 1 | Linearity | Overall F-test | **F = 3261.38** | 0.0 | **PASS** |
| 2 | Independence | Durbin-Watson | **1.9618** | n/a | **PASS** (~2) |
| 3 | Homoscedasticity | Breusch-Pagan | LM = 1170.51 | 2.25e-247 | **FAIL** (heteroscedastic) |
| 4 | Normality | Jarque-Bera | **9371.47** | 0.0 | **FAIL** (non-normal) |
| 5 | Multicollinearity | VIF | max **9.2061** | n/a | **FAIL** (VIF > 5) |

Residual shape: skewness **+1.0706**, excess kurtosis **+6.0061**. Condition number 6.54.
**Assumptions satisfied: 2 of 5.**

*Figures 16-20 | Tables 20, 21*

### Comment and justification

**1. Linearity - PASS.** The overall F-test is overwhelmingly significant (F = 3261, p ~ 0), so a
linear functional form is justified *in its parameters*. This tests significance, not perfect fit;
Figure 16 (residuals vs fitted) is the visual confirmation. Note the distinction: a model can be
statistically significant and still be a poor predictor - training R2 is 0.613 but test R2 falls
to 0.576.

**2. Independence - PASS, with an honest caveat.** DW = 1.9618 is almost exactly 2, indicating
essentially uncorrelated errors. However, the rows are grouped into ~6,000 geographic census
blocks, so independence is *assumed by the OLS model rather than proven by it*. It is reasonable
here because the blocks are spatially interleaved, but a spatial-error or mixed-effects model
would be the rigorous choice for clustered data.

**3. Homoscedasticity - FAIL (p = 2.25e-247).** The residual variance is definitively not constant.
The model is more accurate for mid-priced houses and less accurate at the extremes - exactly the
signature of the ceiling at 5.0 identified in Task 1. Figure 19 (scale-location) shows the
characteristic upward-trending envelope. **Practical fixes:** log-transform the target, use robust
standard errors, or fit a Tobit model for the censoring.

**4. Normality - FAIL (JB = 9371, p ~ 0).** Residuals are strongly non-normal, with skewness +1.07
and excess kurtosis +6.01. Important nuance: with n = 16,512 the Jarque-Bera test has enormous
power, so even a mild departure would be flagged. The S-shaped departure in the Q-Q plot

---

## 10. Task 8 - Feature Significance and Coefficient Interpretation

### Output

| Feature | Beta (per SD) | Std error | t | p | 95% CI | In USD, significant? |
|---|---|---|---|---|---|---|
| MedInc | +0.8544 | 0.0089 | +95.70 | 0.000 | 0.837 - 0.872 | +$85,438 YES |
| Latitude | -0.8969 | 0.0170 | -52.77 | 0.000 | -0.930 - -0.864 | -$89,693 YES |
| Longitude | -0.8698 | 0.0167 | -52.12 | 0.000 | -0.903 - -0.837 | -$86,984 YES |
| AveBedrms | +0.3393 | 0.0144 | +23.56 | 0.000 | 0.311 - 0.367 | +$33,926 YES |
| HouseAge | +0.1225 | 0.0062 | +19.67 | 0.000 | 0.110 - 0.134 | +$12,255 YES |
| AveRooms | -0.2944 | 0.0158 | -18.68 | 0.000 | -0.325 - -0.264 | -$29,441 YES |
| AveOccup | -0.0408 | 0.0056 | -7.25 | 0.000 | -0.052 - -0.030 | -$4,083 YES |
| **Population** | **-0.0023** | 0.0060 | -0.39 | **0.699** | -0.014 - 0.009 | -$231 **NO** |

*Figure 21 | Tables 21, 22*

### Comment and justification

- **MedInc is the strongest positive driver**: a one-standard-deviation rise in district income
  adds about **$85,400** to the median house value. Economically unsurprising but quantitatively
  large, and it also holds the highest Random Forest importance (0.526).

- **Latitude and Longitude are jointly the largest effect in the model** (t = -52.8 and -52.1).
  Location dominates once income is controlled for. The negative Latitude coefficient means value
  falls as one moves *north* up the state. **These two must be read together** - alone, each
  describes a curved coastal strip rather than a direction, which is why they are highly
  collinear (VIF 9.21) yet both remain strongly significant.

- **AveRooms is negative while AveBedrms is positive, and BOTH are individually significant.**
  This is a **suppression effect**: because the two are 0.85 correlated, each absorbs part of the
  other's influence. The economically correct reading is that *room count* adds value once
  household size is controlled for - more rooms in a house occupied by fewer people. **Neither
  sign should be interpreted in isolation.** This is precisely why the VIF analysis in Task 4
  matters: without it, a reader would wrongly conclude that more rooms reduce house value.

- **Population is the only insignificant variable** (p = 0.699). Once MedInc, the coordinates and
  the rooms-per-person ratio are known, the raw block population adds nothing further. It is a
  genuine "droppable" variable, though retained for comparability across models.

- **HouseAge is mildly positive** (+$12,255 per SD) - vintage matters in California, unlike in
  most older housing markets.

- **The contrast with Task 2 is the key lesson:** AveBedrms correlates only -0.047 with the target
  (essentially zero) yet has the **fourth largest coefficient** in the model. It has no marginal
  relationship but a strong *partial* relationship, because it only matters once income and
  location are held constant. Anyone selecting variables from a correlation matrix alone would
  have discarded one of the most important predictors.

---

## 11. Effect of Regularisation on Coefficients

---

## 12. Task 10 - Final Model Selection with Justification

### Decision

| Role | Model | R2 (test) | CV R2 | MAE ($) |
|---|---|---|---|---|
| **PRIMARY - interpretable** | **Elastic Net** (Ridge as close alternative) | 0.5768 | 0.5741 | $53,316 |
| **BENCHMARK - accuracy ceiling** | **Random Forest** | **0.8062** | **0.8045** | **$32,681** |

Random Forest is the best model on *both* the held-out test set and 5-fold cross-validation, and
ranks first on all five metrics. The recommendation is nevertheless a **two-model answer**, because
the brief explicitly requires coefficient interpretation and feature significance - requirements a
tree ensemble structurally cannot satisfy.

### Justification

**1. Accuracy.** Random Forest wins outright: R2 0.8062 vs 0.5768 for the best linear model, an
improvement of 0.23 in explained variance and $20,600 in typical MAE. Confirmed by 5-fold CV
(0.8045), so it is not an artefact of one lucky split.

**2. The failed assumptions force a regularised linear model.** Homoscedasticity, normality and
multicollinearity all fail (Task 9). Plain OLS therefore gives unreliable p-values and confidence
intervals. Ridge/Elastic Net address the variance inflation while preserving a fully readable
equation.

**3. Interpretability is an explicit requirement.** The brief asks for coefficient interpretation
and feature significance. Every linear model yields a readable equation - "a one standard
deviation rise in MedInc adds $85,438" - whereas a Random Forest yields only importance scores with
no direction, magnitude, or significance. For a valuation report that transparency is essential, and
no amount of accuracy compensates for its absence when the deliverable is an explanation.

**4. The nonlinear benchmark is evidence, not the answer.** The 0.23 R2 gap *proves* the true
relationship contains curvature and interactions that no linear or quadratic term captures. The
forest is reported as the achievable ceiling, so the reader knows what the linear model leaves on
the table.

**5. Why Elastic Net rather than Lasso.** Cross-validation gave Lasso a near-zero alpha, so it
selected nothing and collapsed onto OLS. Elastic Net is preferred because it degrades gracefully -
the same l1_ratio dial switches between L1 selection and L2 shrinkage - so if a future resample
pushed alpha higher, it would remain the safer of the two to deploy. Ridge is a perfectly
defensible alternative if simplicity is prioritised.

**6. Why Population was not dropped** despite p = 0.699: retaining it keeps all nine models directly
comparable and the cost is negligible. A production model would drop it.

---


---

## 14. Overall conclusion

All ten mandatory tasks were completed, producing **23 figures and 25 tables** from a single
reproducible script.

House value here is driven jointly by **income (MedInc, +$85,438 per SD)** and **geographic
location (Latitude/Longitude, jointly the largest effect)**, with room count, vintage and occupancy
making smaller but statistically clear contributions. **Population was the only insignificant
variable.**

The most valuable outcome was diagnostic rather than predictive: the dataset violates **three of
five** regression assumptions, and identifying *why* - heteroscedasticity and non-normality trace
directly to the $500,005 censoring ceiling, while multicollinearity traces to the structural
relationship between the coordinates of a coastline - is what justifies the modelling choices. Two
findings are worth carrying forward: **AveBedrms has near-zero marginal correlation but the fourth
largest coefficient**, so correlation-based feature selection would discard a critical predictor;
and **AveRooms carries a negative coefficient that is only interpretable once its collinear partner
is accounted for**.

On model choice the honest conclusion is a **two-part answer**: Random Forest is the accurate model
(R2 0.8062, MAE $32,681) and should be used wherever prediction is the sole objective, while
**Elastic Net (R2 0.5768) is the recommended model for this report** because the brief requires
interpretable coefficients and significance testing, which tree ensembles cannot provide. The
0.23 R2 sacrifice is the price of that interpretability - worth paying when the deliverable is an
explanation rather than a number.

---

## Appendix A - Figure index

| Figure | Content | Task |
|---|---|---|
| 1 | Histograms of all predictors and target | 1 EDA |
| 2 | Boxplots of all variables (outliers in red) | 1 / 3 |
| 3 | Target distribution showing the 5.0 ceiling | 1 EDA |
| 4 | Geospatial maps (value and income) | 1 EDA |
| 5 | Missing values per column (all zero) | 1 EDA |
| 6 | Pearson correlation heatmap | 2 |
| 7 | Scatter plots of every predictor vs target | 2 |
| 8 | Predictor-only correlation matrix | 2 / 4 |
| 9 | Mahalanobis distance outlier detection | 3 |
| 10 | Leverage vs studentised residual (influence plot) | 3 |
| 11 | VIF per predictor with 5 and 10 thresholds | 4 |
| 12 | Simple Linear Regression predicted vs actual | 5 |
| 13 | Multiple Linear Regression predicted vs actual | 5 |
| 14 | Random Forest feature importance | 5 / 7 |
| 15 | Model comparison across R2, RMSE, MAE | 6 |
| 16 | Residuals vs fitted (train and test) | 7 |
| 17 | Normal Q-Q plot of residuals | 7 |
| 18 | Residual distribution vs normal reference | 7 |
| 19 | Scale-location plot | 7 |
| 20 | Actual vs predicted (test set) | 6 / 7 |
| 21 | OLS coefficients with significance | 8 |
| 22 | Effect of regularisation on coefficients | 9 / 10 |
| 23 | Cross-validated model comparison | 10 |

## Appendix B - How to reproduce

```powershell
python run_analysis.py
```

Requires the packages in `requirements.txt`. The script uses `RANDOM_STATE = 42` throughout, so
results are fully deterministic. Runtime is roughly 60 seconds; all figures and tables are written
to `outputs/`.

## 13. Limitations and recommendations

1. **The target is censored at $500,005** - the binding constraint on accuracy. Predictions for
   expensive houses are biased downward, capping linear-model R2 near 0.6. With uncensored prices,
   a Tobit regression would be methodologically correct and would likely improve on every model here.

2. **Independence is assumed, not verified.** The ~6,000 geographic blocks are a clustered
   structure; a spatial-error or mixed-effects model would be more rigorous than plain OLS.

3. **The regularised models are effectively OLS here** (Task 11). Their advantage would appear on a
   smaller sample. This should be stated rather than presented as a successful regularisation exercise.

4. **Gradient Boosting is under-trained** (lr = 0.05, 300 trees). Increasing the estimator count is
   the obvious next step.

5. **One highly influential row** (19006, Cook's D = 0.77) affects mainly the geographic
   coefficients. Any published Latitude/Longitude coefficient should carry this caveat.

6. **No spatial cross-validation.** A random split lets neighbouring blocks appear in both train and
   test, which slightly *inflates* all reported scores. Block-group or spatial k-fold CV would give
   a more honest estimate for geographic data.


### Output

| Feature | OLS | Ridge | Lasso | Elastic Net | Ridge shrink % | Lasso shrink % |
|---|---|---|---|---|---|---|
| MedInc | 0.8544 | 0.8542 | 0.8511 | 0.8516 | 0.03% | 0.39% |
| HouseAge | 0.1225 | 0.1229 | 0.1231 | 0.1230 | -0.25% | -0.41% |
| AveRooms | -0.2944 | -0.2936 | -0.2861 | -0.2873 | 0.27% | 2.81% |
| AveBedrms | 0.3393 | 0.3383 | 0.3309 | 0.3321 | 0.30% | 2.45% |
| Population | -0.0023 | -0.0022 | -0.0015 | -0.0016 | 4.39% | 34.09% |
| AveOccup | -0.0408 | -0.0408 | -0.0402 | -0.0403 | -0.04% | 1.45% |
| Latitude | -0.8969 | -0.8939 | -0.8899 | -0.8906 | 0.34% | 0.78% |
| Longitude | -0.8698 | -0.8668 | -0.8624 | -0.8632 | 0.35% | 0.85% |

*Figure 22 | Table 23*

### Comment and justification

This table is the most instructive negative result in the analysis.

- **No coefficient is meaningfully shrunk.** Ridge reduces every one by under 0.5%, and Lasso
  zeroed nothing. Two coefficients even *increase* marginally (HouseAge, AveOccup), which is
  normal - shrinkage is a joint operation, not eight independent decisions.
- **The largest relative shrinkages fall on the two least important variables** (Population 34%
  under Lasso, AveRooms 2.8%), which is the penalty working as designed: it targets weak, redundant
  predictors first.
- **Interpretation:** regularisation is the right *safeguard* given VIFs of 7-9, but on this
  dataset it is not what produces the accuracy - the large sample relative to the small number of
  predictors is. Had n been 200 instead of 16,512, the penalties would have bitten hard and
  dramatically improved the model. The textbook expectation that "Ridge/Lasso fix
  multicollinearity" is true in principle but simply not binding at this sample size.

(Figure 17) confirms it is genuine, not merely a power artefact. **Crucially, normality is required
for valid confidence intervals, not for the point estimates** - the coefficient values remain
usable, but their intervals should be treated with caution.

**5. Multicollinearity - FAIL (max VIF 9.21).** Latitude and Longitude explain ~89% of each other's
variance, so their individual standard errors are inflated about 3x even though the model as a
whole is well specified. This is the assumption that most directly motivates the regularised models.

### Overall conclusion on assumptions

**Linearity and independence hold comfortably. Homoscedasticity, normality and multicollinearity do
not.** The decisive insight is that these three failures affect the reliability of *confidence
intervals and p-values*, **not the usefulness of the point predictions**. The model still ranks
blocks correctly; it simply cannot promise tight intervals. This is why the regularised linear
models are preferred as the interpretable option, and why the violations are a reason to avoid
naive inference rather than a reason to discard the model.

**Residual statistics (test set):** mean +0.0035 | std 0.7456 | min/max -9.8753 / +4.1484 |
skewness +0.5459 | kurtosis +10.0838 | 8.60% above +1 | 2.96% below -1

*Figures 15-20 | Tables 17-19, 24*

### Comment and justification

- **Random Forest ranks first on every single metric**, so the choice is not metric-dependent.
  Its MAE of **$32,681** means a typical valuation error of about $33,000 - a 38% improvement on
  the best linear model's $53,320.
- **R2 and Adj R2 are almost identical everywhere** (differences < 0.0011), which confirms no
  model is being rewarded merely for having more parameters.
- **MAE is always smaller than RMSE** - the expected relationship, since RMSE squares the errors
  and is therefore dominated by the large misses on the extreme (capped) houses. For a real
  valuation business, MAE is the metric that matters, because it is the *typical* error.
- **Huber is the instructive exception**: it ranks 8th on R2 but 4th on MAE. This is precisely
  the behaviour a robust estimator should show, and it is why R2 alone is an incomplete ranking.
- **CV R2 (0.61) is consistently higher than test R2 (0.576) for the linear models.** Within CV
  the models are refitted on 4/5 of 16,512 = 13,210 rows, so standard errors are marginally
  tighter than on the untouched test set. CV figures are used for *ranking*; the held-out test
  figures are quoted as the honest generalisation estimate.
- **Mean residual ~ 0 confirms OLS unbiasedness.** The slight positive skew and heavy excess
  kurtosis (+10.08) reflect the censored target, consistent with the Jarque-Bera rejection in Task 7.

Rule of thumb: VIF < 5 acceptable | 5 <= VIF < 10 moderate | VIF >= 10 severe

| Feature | VIF | R2 (vs others) | Tolerance | Severity |
|---|---|---|---|---|
| **Latitude** | **9.2061** | 0.8914 | 0.1086 | moderate |
| **Longitude** | **8.8760** | 0.8873 | 0.1127 | moderate |
| **AveRooms** | **7.9172** | 0.8737 | 0.1263 | moderate |
| **AveBedrms** | **6.6092** | 0.8487 | 0.1513 | moderate |
| MedInc | 2.5398 | 0.6063 | 0.3937 | acceptable |
| HouseAge | 1.2373 | 0.1918 | 0.8082 | acceptable |
| Population | 1.1348 | 0.1188 | 0.8812 | acceptable |
| AveOccup | 1.0097 | 0.0096 | 0.9904 | acceptable |

**Iterative removal:** Latitude (9.206) -> AveRooms (7.192) -> HouseAge (1.138) -> all under 5.
**2 predictors must be discarded** before every VIF falls below 5.

*Figure 11 | Tables 11-12*

### Comment and justification

- **No VIF exceeds 10**, so there is no *severe* collinearity, but four variables breach the
  conventional VIF > 5 threshold. This is a **moderate** violation, labelled as such rather than
  overstated.
- **Latitude's VIF of 9.21 means 89.1% of its variance is explained by the other predictors
  alone**, and its standard error is inflated by a factor of about **3.0x** relative to 1.0.
- The offending pairs match Task 2 exactly - the expected consistency between the two diagnostics:
  **Latitude/Longitude** (structural) and **AveRooms/AveBedrms** (size variables).
- **Critical methodological point: multicollinearity does not bias OLS coefficients.** It inflates
  their *standard errors*, so t-statistics, p-values and confidence intervals become unreliable,
  while fitted values and overall R2 remain essentially unaffected. The damage is to
  *interpretation*, not *prediction*.
- **Variables are deliberately NOT dropped permanently.** Two would have to go, and discarding
  Latitude/Longitude would remove the model's ability to express the strongest spatial effect in
  the data. Regularisation (Task 6) is the better remedy: it stabilises estimates while retaining
  all the information.

   foreshadows both the collinearity in Task 4 and the curvature that later justifies trees.

**Partition:** 16,512 train / 4,128 test. Target means 2.0719 vs 2.0550 - close, so the random
split is representative and test R2 is a fair estimate.
