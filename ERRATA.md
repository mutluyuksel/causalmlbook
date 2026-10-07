# Suggested code corrections

Four places in the causal chapters where the R code does something other
than what the text around it says. Each was found by running the book's
code with its own seeds (R 4.5, `MatchIt` 4.7.2, `grf` 2.6.1, `survey` 4.5)
and comparing with an independent implementation. Numbers quoted as
"printed" are the ones on the published site (build of 2025-11-19).

The chapter sources are not in this repository, so this file gives the
replacement code instead of a patch.

## 1. Causal Trees and Forests: the interval labelled AUTOC is the ATE's

```r
# Compute a 95% Confidence Interval for the Area Under the TOC (AUTOC)
autoc_ci <- average_treatment_effect(eval.forest, target.sample = "all")
cat("95% CI for AUTOC:", autoc_ci[1], "+/-", 1.96 * autoc_ci[2], "\n")
## 95% CI for AUTOC: -1.054098 +/- 0.3980702
```

`average_treatment_effect()` returns the average treatment effect and its
standard error, so the line prints the ATE of the evaluation forest. The
AUTOC and its standard error are already in `rate.cate`, printed a few
lines earlier as `1.662433` and `0.194141`:

```r
cat("95% CI for AUTOC:", rate.cate$estimate, "+/-", 1.96 * rate.cate$std.err, "\n")
## 95% CI for AUTOC: 1.662433 +/- 0.3805164
```

Two smaller things in the same chapter. TOC stands for Targeting Operator
Characteristic (Yadlowsky et al.), not "Treatment Overlap Curve". And in
the `causal_forest()` call that lists the defaults, `alpha = 0.05` is
commented as the confidence level; in `grf` it is the maximum imbalance of
a split.

## 2. Selection on Observables: the separate-regressions ATE is right by accident

```r
treated_model <- lm(Y ~ X, subset = (D == 1))
control_model <- lm(Y ~ X, subset = (D == 0))
X_treated_mean <- colMeans(X[D == 1,])
X_control_mean <- colMeans(X[D == 0,])
Y1_hat <- predict(treated_model, newdata = data.frame(t(X_treated_mean)))
Y0_hat <- predict(control_model, newdata = data.frame(t(X_control_mean)))
ATE_separate <- mean(Y1_hat - Y0_hat)
## [1] 2.990066
```

`X` is a matrix in the global environment, and `newdata` has no column
called `X`, so `predict()` ignores `newdata` and warns:

```
'newdata' had 1 row but variables found have 1000 rows
```

`Y1_hat` and `Y0_hat` are therefore predictions for all 1,000 units, and
their mean difference is the regression-adjusted ATE. That is the right
estimator, and 2.990066 is the right number, but it is not what the code
says it does. Had the group means been used as written, the result would
have been the raw difference in means (2.917305): a regression with an
intercept, evaluated at its own sample mean of the covariates, returns its
own sample mean of the outcome.

A version that does what the text describes, with the same output:

```r
df <- data.frame(Y = as.numeric(Y), D = D, X1 = X[, 1], X2 = X[, 2], X3 = X[, 3])
treated_model <- lm(Y ~ X1 + X2 + X3, data = df, subset = D == 1)
control_model <- lm(Y ~ X1 + X2 + X3, data = df, subset = D == 0)
# predict both potential outcomes for every unit, then average
ATE_separate <- mean(predict(treated_model, newdata = df) -
                     predict(control_model, newdata = df))
## [1] 2.990066
```

## 3. Matching Methods: the full-matching estimate ignores the matching weights

```r
estimate_ATT <- function(matched_data) {
  mean(matched_data$Earnings[matched_data$Treatment == 1]) -
  mean(matched_data$Earnings[matched_data$Treatment == 0])
}
cat("ATT (Full Matching):", estimate_ATT(matched_data_full), "\n")
## ATT (Full Matching): 3.61869
```

After 1:1 matching without replacement the unweighted difference is the
estimate. After full matching it is not: matched sets differ in size and
`match.data()` returns the `weights` that account for it. With them:

```r
library(sandwich); library(lmtest)
fit <- lm(Earnings ~ Treatment, data = matched_data_full, weights = weights)
coeftest(fit, vcov. = vcovCL, cluster = ~subclass)["Treatment", ]
##  Estimate Std. Error
##  3.462011   1.013667
```

In the same chapter:

- "Overall Weighted ATT" in the subclassification example weights each
  stratum by its total size, `(N_treat + N_control) / N`. That is the ATE
  (3.184567). Weighting by `N_treat / sum(N_treat)` gives the ATT, 3.156913.
- The block headed "Matching Methods Without Propensity Scores" calls
  `matchit(..., method = "nearest")` and `method = "full"` with the default
  `distance = "glm"`, which is a logit propensity score. Only the
  Mahalanobis call matches on covariates.
- In the XGBoost example the object named `nn_match` and reported as
  "Nearest Neighbor" is fitted with `method = "optimal"`.

## 4. Inverse Weighting: "IPW" and "Hajek IPW" are the same estimator

```r
ipw_estimate   <- svyglm(Earnings ~ Treatment, design = svydesign(~1, weights = ~IPW, data = data))
hajek_estimate <- svyglm(Earnings ~ Treatment, design = svydesign(~1, weights = ~IPW_Hajek, data = data))
## ATT (IPW): 3.219614
## ATT (Hajek IPW): 3.219614
```

A weighted regression on a treatment dummy is a difference of weighted
means, each divided by the sum of its own weights. It is already the
normalised (Hajek) estimator, and rescaling the weights within arm cannot
change it, which is why the two lines agree to every digit. The
Horvitz-Thompson estimator divides by `N` instead and is very different on
these data, because the propensity score ranges from 0.0009 to 0.99:

```r
D <- data$Treatment; Y <- data$Earnings; e <- data$PS
ate_ht    <- mean(D * Y / e) - mean((1 - D) * Y / (1 - e))
ate_hajek <- sum(D * Y / e) / sum(D / e) -
             sum((1 - D) * Y / (1 - e)) / sum((1 - D) / (1 - e))
## ate_ht    2.235377
## ate_hajek 3.219614
```

Both lines are labelled ATT. The weights `1/PS` and `1/(1 - PS)` target
the ATE; ATT weights are 1 for treated units and `PS / (1 - PS)` for
controls.

Also here, `table1_weighted`, printed as "Balance after IPW", is
`svytable(~Treatment)`: the weighted count in each arm. A weighted balance
table would be, for example,
`tableone::svyCreateTableOne(vars, strata = "Treatment", data = svydesign(~1, weights = ~IPW, data = data))`.
