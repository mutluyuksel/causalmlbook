# Double Machine Learning

This chapter builds directly on the previous one, which introduced causal inference under selection on observables using regression adjustment and traditional covariate control methods. While those tools work well in low-dimensional settings with a moderate number of covariates, they face limitations when dealing with high-dimensional data or when model selection is required. In this chapter, we introduce modern, high-dimensional methods—particularly Double LASSO and Double Machine Learning (DML)—that address these challenges by combining regularized estimation with orthogonalization and sample splitting. These techniques enable valid inference for treatment effects even when the number of covariates is large, and they form a core part of the modern causal machine learning toolkit.

We begin by reviewing the key idea behind Double LASSO and partialling out, then explain the logic and mechanics of DML1 and DML2 estimators. Using simulations, we show how these methods correct for regularization bias and maintain valid inference through orthogonal score construction and cross-fitting. Readers will learn how to implement both cross-validated and plug-in penalty approaches for tuning, how to adapt the methods with different learners such as trees and random forests, and how to compute robust standard errors. By the end of the chapter, readers will be able to apply Double Machine Learning tools in high-dimensional causal settings and understand when and why they work.

To estimate causal/treatment effect, we often start with a model of the form

\begin{equation}
Y = \delta \cdot D + f(X) + \varepsilon
\end{equation}

where \(Y\) is the outcome, \(D\) is the treatment, \(X\) is a (potentially large) set of covariates that may confound the relationship between \(D\) and \(Y\), and \(\delta\) is the average treatment effect of interest. While \(f(X)\) could be any function in principle, implementing LASSO typically requires us to assume linear structure so that we can apply penalized linear regression methods in high dimensions.

When \(p\geq n\) (or \(p\) is large relative to \(n\)), ordinary least squares (OLS) can overfit or fail to run due to singular design matrices. LASSO (the \(\ell_1\)-penalized least squares estimator) addresses this by shrinking coefficients toward zero, effectively selecting a smaller set of covariates. This helps manage high-dimensional confounding while reducing overfitting, as previously discussed in the penalized regression methods chapter. However, it also introduces shrinkage bias, which can distort parameter estimates (here, average treatment effect) if left uncorrected. Several techniques we cover in this section tackle the bias problem by combining LASSO estimator, partialling-out procedure, and orthogonalization to achieve debiased estimated parameters with valid inference. 

The Frisch-Waugh-Lovell (FWL) theorem provides the foundation for partialling out, which is central to Double LASSO methods, we will discuss in this section. In a linear model where \( p \leq n \), FWL states that the coefficient on the treatment variable can be obtained by residualizing both the outcome and the treatment on the covariates and then regressing the residualized outcome on the residualized treatment. This procedure effectively estimating the causal/treatment effect,\(\delta\) from the model, \(Y_i- E[Y \mid X] = \delta . (D-E[D \mid X]) +\epsilon\). This ensures that the estimated treatment effect retains the same standard errors and confidence intervals as those obtained from direct OLS regression, making inference straightforward and comparable to traditional methods. We recommend to review the the detailed discussion with simulation in the last section.

Double LASSO methods extend this concept to high-dimensional settings (\( p \geq n \)), where selecting relevant covariates is crucial to avoid overfitting thus decreasing sahrinkage bias while still permitting valid inference. These methods are also used for causal or treatment-effect estimation. They include Double LASSO with partialling out, Double LASSO with double selection, and a more recent approach Orthogonal/Double Machine Learning LASSO (DML1 and DML2). While cross-validated DML2 is often recommended in high-dimensional applications (Chernozhukov et al. (2018)), it can be computationally intensive. In practice, other Double LASSO approaches remain valid and efficient for estimating \(\delta\). Below, we outline the main variations of Double LASSO and how we can implement to estimate average causal/treatment effect, \(\delta\).

We want to warn readers that while LASSO is useful for penalized estimation in sparse models, helping to identify variables strongly associated with an outcome and improving prediction accuracy. However, naively using LASSO for inference on model parameters can be problematic since it is designed for prediction, not hypothesis testing. To ensure valid inference, variable selection should be applied separately to the reduced-form equations (outcome and treatment models), and all selected variables should be included in the final OLS estimation to reduce omitted variable bias when estimating causal effects. Also, Using LASSO for inference may require a different penalty selection (\(\lambda\)) than what is optimal for prediction, which we will cover the plug-in penalty section below.


**Double LASSO – Partialling-Out Model**

A popular approach to estimate \(\delta\) in the partially linear setting relies on partialling out (Belloni, Chernozhukov, and Hansen, 2014a). Let

\begin{equation}
Y_i = X_i^\top \gamma + \tilde{Y}_i, 
\quad
D_i = X_i^\top \pi + \tilde{D}_i
\end{equation}

where \(\hat{\gamma}\) and \(\hat{\pi}\) are coefficients estimated via LASSO regressions of \(Y\) on \(X\) and \(D\) on \(X\), respectively. The residuals 

\begin{equation}
\tilde{Y}_i = Y_i - X_i^\top \hat{\gamma}, 
\quad
\tilde{D}_i = D_i - X_i^\top \hat{\pi}
\end{equation}

represent the variation in \(Y\) and \(D\) that is not explained by \(X\). The key insight is that once we remove the portion of \(Y\) and \(D\) explained by \(X\), a simple regression of \(\tilde{Y}_i\) on \(\tilde{D}_i\) identifies the average treatment effect, \(\delta\):

\begin{equation}
\tilde{Y}_i = \delta \,\tilde{D}_i + \xi_i
\end{equation}

This procedure is sometimes called “partialling out” because it resembles running a regression of \(Y\) on \(D\) and \(X\) but in two steps: first, partial out \(X\) from \(Y\) and \(D\), then regress the residuals. The advantage is that LASSO handles the high-dimensional \(X\) in each of these auxiliary regressions. A limitation is that LASSO may exclude variables with small but nonzero coefficients or struggle in highly correlated designs. Nevertheless, the final residuals \(\tilde{Y}_i\) and \(\tilde{D}_i\) still isolate the variation needed to estimate \(\delta\).


**Double LASSO – Double Selection Model**

In the Double Selection version (Belloni, Chernozhukov, and Hansen, 2014b), the first two steps—regressing \(Y\) on \(X\) and \(D\) on \(X\)—remain the same. However, in the final step, one includes the selected covariates from both LASSO regressions alongside \(\tilde{D}_i\). Formally, let \(X_{\text{selected}}\) be the union of covariates chosen by LASSO in the two separate regressions; the final model becomes

\begin{equation}
\tilde{Y}_i = \delta \,\tilde{D}_i + X_{\text{selected}}^\top \theta + \xi_i
\end{equation}

Here, \(\tilde{Y}_i\) is again the residual of \(Y\) on \(X\), and \(\tilde{D}_i\) is the residual of \(D\) on \(X\). Including all selected covariates ensures that any predictor deemed important for \(Y\) or \(D\) is present in the final stage, mitigating omitted-variable bias and preserving valid standard errors.

Whereas partialling out conceptually resembles a direct regression of \(\tilde{Y}\) on \(\tilde{D}\), double selection is closer to an OLS of \(Y\) on \(D\) plus the combined set of selected covariates from the regression adjustment section. Both are variations on the Double LASSO framework and can be used to obtain a consistent estimate of the average treatment effect, \(\delta\).


## Double Machine Learning 

Standard LASSO estimates are biased due to the \(\ell_1\)-penalty shrinking coefficients toward zero. Debiased LASSO (also known as desparsified or de-sparsified LASSO) corrects this shrinkage by adding a carefully constructed adjustment term, pioneered by Zhang and Zhang (2014), van de Geer et al. (2014), and Javanmard and Montanari (2014). In essence, Debiased LASSO applies a correction term to the initial LASSO estimates to remove shrinkage bias. While it can be used for inference across all coefficients, it focuses on correcting specifically only LASSO’s penalty-induced bias.

A more recent and generalized approach to debiase LASSO estimates is Double Machine Learning (DML) (Chernozhukov et al. 2018). Although Double Machine Learning can incorporate various machine-learning methods (e.g., random forests, gradient boosting), we emphasize LASSO in this section, then discuss decision trees and random forest. Debiased LASSO corrects the shrinkage bias in a single high-dimensional linear model, whereas Orthogonal/DML frameworks explicitly account for potential causal structures (e.g., treatment, outcome, high-dimensional controls) and often use sample splitting and cross-fitting to bolster inference. We highly recommend consulting the last section for a reader-friendly explanation of the orthogonalization idea, as well as more theoretical details of DML.  

DML is designed for estimating causal parameters in high-dimensional settings and is built around orthogonalized estimating equations, which safeguard the final estimate against small errors in the nuisance functions (the parts of the model not containing the main parameter of interest; for example, \(f(X)\)). When using LASSO specifically, we fit high-dimensional regressions to learn these nuisance functions (here \(E[Y \mid X]\) and \(E[D \mid X]\), estimated in the second step), then orthogonalize the residuals (implementing the third step) to ensure that the final treatment effect estimate is unbiased. Meanwhile, DML applies sample splitting (cross-fitting) to ensure that the data used to learn \(\hat{Y}(X)\) and \(\hat{p}(X)\) are not reused for the regression of the residuals. By doing so, bias from overfitting is mitigated, and valid inference on \(\delta\) is obtained.

Several software packages implement Double Machine Learning (DML) across Stata, R, and Python. In Stata, the most common tools include `lasso` (with `selection(plugin)`), `ds` (double selection), `pdslasso` (Post-Double Selection Lasso), and `dml` (for DML1 and DML2 estimators). In R, key packages include `hdm`, which provides `rlasso()` for plug-in penalty selection, `DoubleML`, which supports various ML models including Lasso, and `grf` for causal forests. In Python, `DoubleML` mirrors the R implementation, offering DML1, DML2, and IV-DML with Lasso, Random Forests, and Boosting, while `econml` from Microsoft provides flexible causal ML models, including Lasso-based DML.  

In this section, we will cover DML with Lasso in great detail, building our own simulation from the ground up. While we will eventually use `hdm::rlasso()`, our approach will first provide step-by-step explanations, covering default settings of different programs, theoretical foundations, and practical considerations. This ensures that readers fully grasp the mechanics of DML before applying prebuilt implementations. By deeply understanding the method, readers will find it easier to extend DML to IV estimation, Random Forest-based DML, and even Deep Learning-based DML. Additionally, as DML is an evolving field, this foundation will enable readers to follow the latest research and updates in the literature with confidence. 

### Brief Practical Framework (DML1 & DML2 for LASSO)

  1. **Split** the sample into \(K\) folds.  

   **For each fold \(k\):**
   
  2a. **Use \(K-1\) folds** to estimate the nuisance functions:
   - **Fit** \(\hat{Y}_{-k}(X)\) by running LASSO or another ML method to predict \(Y\).
   
   - **Fit** \(\hat{p}_{-k}(X)\) by running LASSO or another ML method to predict \(D\).
   
   - **Select** the "best" tuning parameter \(\lambda\).
   
When using LASSO, the tuning parameter \(\lambda\) can be selected via cross-validation (will show in first simulation below), plug-in method (i.e. direct calculation with formula based on Belloni, Chernozhukov, and Hansen (2014); will discuss and show in another simulation below).

  2b. In fold \(k\):

   - **Compute** cross-fitted residuals:
   
     \begin{equation}
     \tilde{Y}_i = Y_i - \hat{Y}_{-k}(X_i), \quad
     \tilde{D}_i = D_i - \hat{p}_{-k}(X_i)
     \end{equation}
  3. **Estimate** the treatment effect \(\delta\) by regressing \(\tilde{Y}_i\) on \(\tilde{D}_i\):

   - In **DML1**: Perform this OLS fold by fold, producing one \(\delta\) estimate per fold, then average across folds.
   
   - In **DML2**: Pool the residuals from all folds first, then run a single OLS of \(\tilde{Y}\) on \(\tilde{D}\).

This approach balances flexibility (any suitable ML can be used) with reliable inference on \(\delta\). The orthogonal score structure ensures that moderate errors in the estimated nuisance functions \(\hat{Y}(X)\) and \(\hat{p}(X)\) do not contaminate the final treatment effect estimate.

Alternative methods such as ridge regression, adaptive LASSO, and the elastic net can also be employed in nuisance function estimation (in Step 2). Adaptive LASSO modifies penalty weights to reduce bias, while elastic net combines the LASSO penalty with a ridge component, improving performance in the presence of highly correlated covariates. Other related techniques, such as the Dantzig selector and square-root LASSO, provide additional flexibility depending on the specific data structure and model requirements. Moreover, in both the Double LASSO - Partialling-Out Model and the Double LASSO - Double Selection Model, Post-LASSO can be applied to refit OLS on the covariates selected by LASSO, reducing shrinkage bias (in nuisance function estimation) while maintaining selection advantages.[^PostLas] The next parts of this chapter—covering standard errors, step-by-step implementation for DML1 and DML2, and simulation—will help you see these methods in action.

[^PostLas]:Post-LASSO is implemented by first running LASSO for variable selection and then refitting an unpenalized OLS regression on the selected covariates. In the Partialling-Out Model, this applies to the regressions of \( Y \) and \( D \) on \( X \) before computing residuals. In the Double Selection Model, Post-LASSO is applied separately in the first two steps before the final OLS regression of \( Y \) on \( D \) and the selected covariates.

### Inference and Robust Standard Errors

For LASSO to perform effectively in these settings, two key conditions must be met: approximate sparsity and the irrepresentable condition. Approximate sparsity ensures that only a small subset of covariates significantly influence the outcome and treatment models, allowing LASSO to focus on the most relevant variables. The irrepresentable condition guarantees that the selected covariates are not excessively correlated with the unselected ones, preventing LASSO from mistakenly omitting important variables or selecting irrelevant ones due to high collinearity. These conditions are fundamental for LASSO to consistently identify the relevant covariates and produce accurate treatment effect estimates. In practice, researchers assess these assumptions to ensure the reliability and validity of LASSO-based estimates. We will outline specific conditions and methods for testing them after presenting the simulation results.  

Across all three methods, the final step involves an OLS regression of residuals, which is equivalent to a standard regression of the treatment effect. The asymptotic distribution of this estimator follows:  

\begin{equation}
\hat{\beta} \sim N\left(\beta, \widehat{\text{Var}}(\hat{\beta})\right)
\end{equation}

where the variance is estimated using a heteroskedasticity-robust approach. In practice, this variance estimator is computed using Eicker-Huber-White (HC3/EHW) robust standard errors[^DMLstd].[^Deblas]

[^DMLstd]:In DML (Chernozhukov et al., 2018), \( \hat{\omega}_i \) is derived from the Neyman orthogonality condition, ensuring that small errors in nuisance function estimation do not affect the final treatment effect estimate. The final variance of \( \hat{\beta} \) is estimated using a heteroskedasticity-robust formula that incorporates both the residual variation and the influence function correction term:  
\[
\widehat{\text{Var}}(\hat{\beta}) = \left(\frac{1}{n} \sum_{i=1}^{n} \hat{\omega}_i^2 \right)
\]
where \( \hat{\omega}_i \) represents the influence function correction term that adjusts for regularization bias. The corresponding heteroskedasticity-robust standard error is given by:
\[
\widehat{\text{SE}}(\hat{\beta}) = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \hat{\omega}_i^2 }
\]
In practice, this variance estimator is computed using Eicker-Huber-White (EHW) robust standard errors, but with additional corrections to account for the impact of machine learning-based nuisance function estimation. The "additional corrections" in DML come from using cross-fitting and Neyman orthogonality, which remove bias from nuisance function estimation. This adjustment ensures that the inference for \( \hat{\beta} \) remains valid despite using LASSO or other regularization techniques for covariate selection and treatment assignment modeling.

[^Deblas]:To estimate \( \hat{\omega}_i \) in Debiased LASSO (Zhu \& Hastie, 2004; Javanmard \& Montanari, 2014), the correction is done using a nodewise LASSO regression to estimate the inverse covariance matrix of the selected variables. This provides an estimate of how much LASSO’s shrinkage affects inference.

This robust variance estimator (EHW) enables the construction of confidence intervals:

\begin{equation}
\hat{\beta} \pm z_{\alpha/2} \sqrt{\widehat{\text{Var}}(\hat{\beta})}
\end{equation}

and facilitates hypothesis testing using the test statistic:

\begin{equation}
t = \frac{\hat{\beta}}{\sqrt{\widehat{\text{Var}}(\hat{\beta})}}
\end{equation}

which follows an approximate standard normal distribution under the null hypothesis \( H_0: \beta = 0 \). 

### Step-by-step DM1-LASSO and Simulation

Now, we will discuss the step-by-step procedure for DML1-lasso and DML2-lasso, which we believe will provide a deeper understanding of the details before proceeding to the simulation. Lets assume we have a dataset with \( n = 10,000 \) observations and \( p = 100 \) covariates, and also choose 5 fold for outer loop, and 10 fold for inner loop.

**Step-by-step procedure for DML1-LASSO**:

**Step 1: K-Fold Splitting for Cross-Fitting (Outer Loop, K=5)**

- **Partition the dataset into \( K = 5 \) disjoint folds** (each with \( n = 10,000/5 = 2,000 \) observations).

- For each **fold \( k \)**:

  - **Training data**(\( I_k^c \)): The **remaining 8,000** observations.
  
  - **Validation data**(\( I_k \)): The **2,000** left-out observations.
  
**Step 2: Estimating the Nuisance Functions (on Training Data, \( I_k^c \))**
We estimate two separate Lasso regressions in each training set:

**(A) Predict \(Y\) (Outcome Model) (Lasso with Inner Cross-Validation, \( K=10 \))**

- **Partition the training set (8,000 obs) into \( K = 10 \) inner folds**.

- **For each inner fold \( j \) (Nested CV, \( K=10 \))**:

  - Train Lasso regression on **7,200 observations** (training within training); Validate on the left-out **800 observations**.
  
  - Choose the "best" \(\lambda\) that minimizes prediction error (e.g., MSE).
  
- **Train final Lasso model with the best \(\lambda\) on all 8,000 training observations**.

- Compute predictions for the **2,000 validation observations** using:

  \[
  \hat{Y}_k(X) = \hat{\beta}' X
  \]
  
- This gives an estimate of the expected outcome for each individual without treatment effects (potential outcome in control state).

**(B)  Predict \(D\) (Propensity Score Model) (Lasso with Inner Cross-Validation, \( K=10 \))**

- Follow the **same nested cross-validation procedure** as for the outcome model, but now for predicting \( D \):

  \[
  \hat{p}_k(X) = P(D=1 | X)
  \]
  
- This helps control for selection bias in treatment assignment.

*At the end of Step 2**, we have: \( \hat{Y}_k(X) \): The predicted outcome for the validation data.
and \( \hat{p}_k(X) \): The estimated probability of treatment for the validation data.

**Step 3: Estimating \( \hat{\gamma}_{0,k} \) (On Validation Data, \( I_k \))**

For each **outer fold \( k \) (on the 2,000 validation observations)**:

- Compute the residuals:

  \[
  \tilde{Y} = Y - \hat{Y}_k(X), \quad \tilde{D} = D - \hat{p}_k(X)
  \]
  
- Estimate **\( \hat{\gamma}_{0,k} \)** by running the residualized regression:

  \[
  \tilde{Y} = \hat{\gamma}_{0,k} \tilde{D} + \text{error}
  \]

**At the end of Step 3**, we obtain **5 different ATE estimates** \( \hat{\gamma}_{0,k} \) (one per fold).

**Step 4: Aggregating the Estimates**

- Compute the final **cross-fitted ATE** by averaging over the 5 folds:

  \[
  \tilde{\gamma}_0 = \frac{1}{5} \sum_{k=1}^{5} \hat{\gamma}_{0,k}
  \]
  
### Simulation for DML1:

In this simulation we generate a dataset with an outcome, a treatment variable, and a set of covariates. The treatment is assigned based on a logistic model using the covariates, and the outcome is determined by both the treatment effect and the covariates, plus random noise. We then split the data into five folds for cross-fitting. For each fold, we use the remaining data to fit Lasso regressions that predict the outcome and the treatment probability. These predictions are used to calculate residuals, which serve as inputs for an OLS regression to estimate the treatment effect within that fold. Finally, we average the fold-specific estimates to obtain the overall average treatment effect and compute robust standard errors for inference.


``` r
# Load required libraries
library(glmnet)      # For Lasso regression
library(MASS)        # For generating multivariate normal data
library(caret)       # For cross-validation and data partitioning
library(estimatr)    # For robust standard errors

# Set seed for reproducibility
set.seed(123)

# === Step 1: Generate Simulated Data (Y, D, X) ===
N <- 10000  # Number of observations
p <- 10     # Number of covariates

# Generate covariates X ~ N(0, I)
X <- mvrnorm(N, mu = rep(0, p), Sigma = diag(p))

# Generate treatment variable D using a logistic model
D_prob <- plogis(X %*% runif(p, -1, 1))  # Compute propensity scores
D <- rbinom(N, 1, D_prob)  # Assign treatment based on propensity

# Generate outcome Y with treatment effect
gamma_true <- 2  # True treatment effect
beta <- runif(p, -1, 1)  # True coefficients for X
Y <- gamma_true * D + X %*% beta + rnorm(N)  # Outcome model

# === Step 2: Split Data into K=5 Folds ===
K <- 5
folds <- createFolds(1:N, k = K, list = TRUE)

# Initialize storage for residuals and ATE estimates
Y_resid <- rep(NA, N)
D_resid <- rep(NA, N)
ate_k <- rep(NA, K)

# Step 3: Cross-Fitting to Estimate Nuisance Functions and Compute ATE Per Fold 
for (k in 1:K) {
  
  # Training and validation split
  train_idx <- unlist(folds[-k])  # Use all but k-th fold for training
  val_idx <- folds[[k]]  # Use k-th fold for validation

  # Training Data
  X_train <- X[train_idx, ]
  Y_train <- Y[train_idx]
  D_train <- D[train_idx]

  # Validation Data
  X_val <- X[val_idx, ]

  # ---- Step 3A: Fit Lasso for Outcome Model (Y ~ X) ----
  # Cross-validation to choose lambda
  lasso_y <- cv.glmnet(X_train, Y_train, alpha = 1, nfolds = 10)  
  # Predict on validation set
  Y_hat_val <- predict(lasso_y, X_val, s = "lambda.min")  

  # ---- Step 3B: Fit Lasso for Propensity Model (D ~ X) ----
  # Logistic Lasso
  lasso_d <- cv.glmnet(X_train, D_train, alpha = 1, family = "binomial", nfolds=10)  
  # Predict propensity
  D_hat_val <- predict(lasso_d, X_val, s = "lambda.min", type = "response")  
# alpha = 0 for Ridge, alpha = 1 for Lasso, alpha = 0.5 for Elastic Net
  # ---- Compute Residuals ----
  Y_resid[val_idx] <- Y[val_idx] - Y_hat_val
  D_resid[val_idx] <- D[val_idx] - D_hat_val

  # ---- Step 3C: Estimate ATE for This Fold Using OLS ----
  ate_k[k] <- coef(lm(Y_resid[val_idx] ~ D_resid[val_idx] - 1))[1]  # No intercept
}

# === Step 4: Compute Final ATE by Averaging Over Folds ===
ate_dml1 <- mean(ate_k)

# === Step 5: Compute Standard Errors Using HC3 Robust SEs ===
dml1_model <- lm_robust(Y_resid ~ D_resid - 1, se_type = "HC3")  #Uses all residuals

# Print results
cat("Estimated ATE using DML1:", ate_dml1, "\n")
```

```
## Estimated ATE using DML1: 1.994326
```

``` r
cat("Standard Error (HC3):", dml1_model$std.error[1], "\n")
```

```
## Standard Error (HC3): 0.023927
```

``` r
cat("95% CI:", dml1_model$conf.low[1], "to", dml1_model$conf.high[1], "\n")
```

```
## 95% CI: 1.947591 to 2.041395
```


### Step-by-step procedure for DML2-LASSO:

**Step 1**,  **Step 2** are same.

**Step 3: Compute Residuals and Estimate \( \tilde{\theta}_0 \) (DML1 vs. DML2 Difference)**

**For each fold \( k \) (on the 2,000 validation observations):**

- Compute residualized \( Y \) (**regression-adjusted outcome**):

  \[
  \tilde{Y}_i = Y_i - \hat{Y}_k(X_i), \quad \text{for } i \in I_k
  \]
  
- Compute residualized \( D \) (**regression-adjusted treatment**):

  \[
  \tilde{D}_i = D_i - \hat{p}_k(X_i), \quad \text{for } i \in I_k
  \]
  
**At the end of Step 3, we have 10,000 residuals for \( Y \) and \( D \), covering the full dataset.**

**Step 4: Estimate the Final ATE**

(Keep in mind **DML1:** Estimate ATE separately for each fold (\( \hat{\gamma}_{0,k} \)), then **average over the 5 folds**.)

- **DML2:** Estimate ATE in one step, using the **full set of 10,000 residuals**:

  \[
  \tilde{Y} = \tilde{\theta}_0 \tilde{D} + \text{error}
  \]
  
**In both methods, the final ATE estimate comes from regressing residualized \( Y \) on residualized \( D \), but DML1 does it per fold before averaging, while DML2 does it in one step using all residuals at once.** 


### Simulation for DML2: 

This simulation also creates a dataset with an outcome, treatment, and covariates. Like before, the treatment is generated using a logistic model and the outcome is a function of both treatment and covariates with added noise. Here, instead of tuning the Lasso model within each fold, we determine an optimal lambda with cross-validation (`cv.glmnet` and `lambda.min` which we explained in detailed in lasso chapter) from the full training data for both the outcome and treatment models. The data is then divided into five folds, and in each fold the fixed lambda is used to predict the outcome and treatment probabilities. Residuals are computed and combined to estimate the treatment effect with robust standard errors. The results from DML2 yield an ATE that is very close to the one from DML1, and DML2 is generally recommended for its simplicity and consistency.



``` r
# Load required libraries
library(glmnet)      # For Lasso regression
library(MASS)        # For generating multivariate normal data
library(caret)       # For cross-validation
library(estimatr)    # For HC3 robust SEs (lm_robust)
set.seed(123) # Set seed for reproducibility

# === Step 1: Generate Simulated Data (Y, D, X) ===
N <- 10000  # Number of observations
p <- 10     # Number of covariates

# Generate covariates X ~ N(0, I)
X <- mvrnorm(N, mu = rep(0, p), Sigma = diag(p))

# Generate treatment variable D using a logistic model
D_prob <- plogis(X %*% runif(p, -1, 1))  # Compute propensity scores
D <- rbinom(N, 1, D_prob)  # Assign treatment based on propensity

# Generate outcome Y with treatment effect
gamma_true <- 2  # True treatment effect
beta <- runif(p, -1, 1)  # True coefficients for X
Y <- gamma_true * D + X %*% beta + rnorm(N)  # Outcome model

# === Step 2: Define Fold Assignments (Standardized) ===
K <- 5
folds <- createFolds(1:N, k = K, list = TRUE)

# === Step 3: Compute a Fixed `lambda.min` Using Full Data ===
cv_lasso_y <- cv.glmnet(X, Y, family = "gaussian", alpha = 1, nfolds = 10)  #for Y
lambda_fixed_y <- cv_lasso_y$lambda.min  # Store optimal lambda for outcome model

cv_lasso_d <- cv.glmnet(X, D, family = "binomial", alpha = 1, nfolds = 10)  #for D
lambda_fixed_d <- cv_lasso_d$lambda.min  # Store optimal lambda for propensity model

# === Step 4: Cross-Fitting with Fixed Lambda ===
Y_resid <- rep(NA, N)
D_resid <- rep(NA, N)

for (k in 1:K) {
  
  # Training and validation split
  train_idx <- unlist(folds[-k])  # Use all but k-th fold for training
  val_idx <- folds[[k]]  # Use k-th fold for validation

  # Training Data
  X_train <- X[train_idx, ]
  Y_train <- Y[train_idx]
  D_train <- D[train_idx]

  # Validation Data
  X_val <- X[val_idx, ]

  # ---- Fit Lasso for Outcome Model (Y ~ X) with Fixed Lambda ----
  lasso_y <- glmnet(X_train, Y_train, family = "gaussian", alpha = 1, 
                    lambda = lambda_fixed_y)
  Y_hat_val <- predict(lasso_y, X_val)

  # ---- Fit Lasso for Propensity Model (D ~ X) with Fixed Lambda ----
  lasso_d <- glmnet(X_train, D_train, family = "binomial", alpha = 1, 
                    lambda = lambda_fixed_d)
  D_hat_val <- predict(lasso_d, X_val, type = "response")

  # ---- Compute Residuals ----
  Y_resid[val_idx] <- Y[val_idx] - Y_hat_val
  D_resid[val_idx] <- D[val_idx] - D_hat_val
}

# === Step 5: Estimate ATE Using HC3/EHW robust SE ===
dml2_model_robust <- lm_robust(Y_resid ~ D_resid, se_type = "HC3")

# === Step 6: Print Results ===
cat("\nEstimated ATE using DML2:", coef(dml2_model_robust)[2], "\n")
```

```
## 
## Estimated ATE using DML2: 1.99448
```

``` r
cat("Standard Error (lm_robust):", dml2_model_robust$std.error[2], "\n")
```

```
## Standard Error (lm_robust): 0.02392815
```

``` r
cat("95% CI (lm_robust):", dml2_model_robust$conf.low[2], "to",
    dml2_model_robust$conf.high[2], "\n")
```

```
## 95% CI (lm_robust): 1.947576 to 2.041384
```



### DML2-Lasso with the Plug-In Penalty

Above, we implemented DML1 and DML2 step by step using cross-validation (CV) to select the tuning parameter \( \lambda \) in Lasso. Cross-validation remains a widely used approach, particularly in prediction-focused machine learning applications. It is computationally feasible even with large datasets, and we fully understand its implementation, having covered all details earlier in penalized regressions-lasso chapter.

However, in causal inference, our goal is not merely to achieve high predictive accuracy but to obtain valid statistical inference (i.e., correct confidence intervals and standard errors). The cross-validation approach is not necessarily always optimal for inference because it minimizes prediction error, whereas in our setting, we need a tuning parameter that ensures valid post-selection inference in high-dimensional models.

The **plug-in penalty method** provides a theory-driven alternative to cross-validation for selecting \( \lambda \). Instead of relying on data-driven minimization of prediction error, the plug-in approach selects \( \lambda \) based on probability bounds that control the impact of noise in high-dimensional settings. This ensures that the resulting Lasso estimator achieves the right level of sparsity while preserving valid inference properties.

The plug-in method is default in Stata’s `lasso` with `selection(plugin)` and R’s `hdm::rlasso()` function. Thus, readers should be familiar with how these packages select the tuning parameter when using Lasso for causal inference.

**How the Plug-In Method Works**

The plug-in approach follows the theoretical framework proposed by Belloni, Chernozhukov, and Hansen (2012, 2014, REStud) and ensures that \( \lambda \) satisfies high-dimensional probability bounds, limiting the influence of noise. Remember, in standard Lasso, we solve: \(\min_{\beta} \frac{1}{2n} \| Y - X \beta \|_2^2 + \lambda \|\beta\|_1\).   The penalty hyper-parameter \( \lambda \) balances fit and sparsity. The challenge is choosing \(\lambda\) so that you retain the true signals while shrinking or eliminating noise variables. Cross-validation chooses \( \lambda \) by minimizing out-of-sample prediction error, while the plug-in method selects \( \lambda \) based on a theoretical noise bound. In a high‑dimensional model (where the number of predictors \( p \) may be large relative to \( n \)), results from high‑dimensional probability tell us that the maximum of the empirical correlations between the predictors and the noise behaves roughly like \(\sqrt{2\log(p)/n}\). To control for this, Belloni, Chernozhukov, and Hansen (equation 2.13, 2014, REStud) sets the penalty level \(\lambda\) on the order of:

   \begin{equation}
   \lambda \asymp c \, \hat{\sigma} \sqrt{n} \, \Phi^{-1}\left(1 - \frac{\gamma}{2p}\right)
   \end{equation}
   
   where \( \hat{\sigma} \) is an estimator of the noise standard deviation; \( \Phi^{-1}(\cdot) \) is the inverse normal CDF (finds a critical value from the standard normal distribution), ensuring the probability bound holds; \(\gamma\) is a small significance level (often chosen based on theory); \( c \) is a constant (default 1.1) to ensure the bound holds with high probability; \( n \) is the number of observations, to control sample size effects; \( p \) is the number of covariates, to account for multiple testing in high dimensions; \( \gamma \) is a small significance level, controlling false positives (default as 0.1).

Unlike cross‑validation, which minimizes prediction error, this theoretical approach targets the property of valid inference after variable selection. This is crucial for applications where we care not only about prediction but also about the correctness of subsequent statistical tests and confidence intervals. Plug-In Method also avoids the computational cost of cross-validation, making it ideal for large datasets. Guarantees valid inference by ensuring \( \lambda \) controls the probability of false selection. Matches the behavior of Stata (`selection(plugin)`) and R (`hdm::rlasso()`), aligning with widely used implementations.

After the initial Lasso selection using the theory-driven \( \lambda \), `rlasso` often performs a “post-Lasso” refit. This means that once the relevant variables are selected, the model is refitted using OLS on the selected variables, reducing the bias introduced by Lasso’s shrinkage. In some cases, an additional adjustment is made by incorporating penalty loadings, which introduce covariate-specific penalization based on heteroskedasticity and non-Gaussian error structures[^Penload]. The inclusion of penalty loadings refines variable selection and can be iteratively updated in an Iterated Lasso procedure[^LAspro]. However, we do not discuss penalty loadings in detail here, as they go beyond the scope of this section. Readers interested in these refinements can refer to equation 2.13 and Appendix Algorithm 1 of the Belloni, Chernozhukov, and Hansen (2014, REStud) paper.

[^Penload]:Penalty loadings modify the standard plug-in formula by assigning different penalty intensities to different covariates, ensuring that shrinkage is applied more appropriately across variables. The adjusted penalty formula includes a diagonal matrix of penalty loadings, where each loading \( \hat{l}_j \) is estimated as \( \hat{l}_j = \sqrt{E_n[x_{ij}^2 \epsilon_i^2]} \). See Belloni, Chernozhukov, and Hansen (2014, REStud) for details.

[^LAspro]:The Iterated Lasso method refines the penalty loadings over multiple steps to ensure that the sparsity structure properly accounts for heteroskedasticity. This approach is implemented through Algorithm 1 in the Appendix of Belloni, Chernozhukov, and Hansen (2014, REStud). The algorithm updates the penalty loadings using the Post-Lasso estimator and stops when convergence criteria are met.

### Simulation for DML2 with Plug-In Penalty

Here we generated the same data as above ( but normally we upload our data) and applied the DML2 method using the plug-in penalty method for selecting the tuning parameter \( \lambda \) in Lasso. We called `hrm` package and used the `hdm::rlasso()` function, which implements the plug-in penalty method by default.  We Fit Lasso for Outcome Model (Y ~ X) , and Fit Lasso for Treatment Model (D ~ X). Here instead of cross-fitting (k=10 fold); using data (all raw data but Kth fold) package calculated \(\lambda\) ; then used post lasso as last step, then  obtain cross-fitted residuals using k-th fold. It does the same process using all 5 fold. Then combine all residuals, and implemented regular OLS with cross-fitted residuals, and EHW robust standard errors.  

We generate the same data as in the previous section (though normally, we would upload our dataset) and apply the DML2 method using the plug-in penalty for selecting the tuning parameter \( \lambda \) in Lasso. We call the `hdm` package and use the `hdm::rlasso()` function, which implements the plug-in penalty method by default. This approach ensures valid post-selection inference without relying on cross-validation.  

**Step 1**: same as DML2 with cross-validation.  
   - The dataset is split into **\( K = 5 \) folds**, each with **\( 2,000 \) validation samples** and **\( 8,000 \) training samples**.  
   - Training data: **\( I_k^c \) (all but the K-th fold)** is used for estimating nuisance functions.  

**Step 2: **Compute the Plug-In Penalty \( \lambda \) (on Training Data, \( I_k^c \))**  
   - The **plug-in formula** for \( \lambda \) is applied using the training data. This calculation **does not depend on Lasso fitting**. It is computed directly from the **training sample**
 **Fit Lasso for Outcome Model (\( Y \sim X \)) Using Plug-In \( \lambda \)**  
   - Use **only the training data \( I_k^c \)**.
   - Apply **Lasso regression with the precomputed plug-in \( \lambda \)**.computed using only the training data
   - Select relevant variables by shrinking some coefficients to zero.

 **Fit Lasso for Treatment Model (\( D \sim X \)) Using Plug-In \( \lambda \)**  
   - Again, use the **same precomputed \( \lambda \)** (computed in Step 2).
   - Select relevant variables and estimate \( \hat{D} \).

 **Post-Lasso Refitting (on Training Data, \( I_k^c \))**  
   - After Lasso selects variables, refit **OLS** using only the selected variables (i.e. calculate OLS coefficients for selected variables).
  
**Step 3** **Predict on Validation Data (\( I_k \))**  
   - Use the **post-lasso refitted model** (OLS coefficients for selected variables) to predict \( \hat{Y} \) and \( \hat{D} \) on the **\( 2,000 \) validation samples**.
   - Compute cross-fitted residuals:
   
   \[
     \tilde{Y} = Y - \hat{Y}, \quad \tilde{D} = D - \hat{D}
   \]
     
**At the end of Step 3, we have 10,000 residuals for \( Y \) and \( D \), covering the full dataset.**

**Step 4:** same as DML2 with cross-validation.
   - A regular OLS regression is performed using the cross-fitted residuals.  
   - Standard errors are computed using EHW robust standard errors to account for heteroskedasticity.  




``` r
# RUN until Step 2 from DML1-CV simulation above to and then run this code
library(hdm)         # For direct rlasso implementation
# === Step 2: Define Fold Assignments (Cross-Fitting) ===
K <- 5
folds <- createFolds(1:N, k = K, list = TRUE)

# === Step 3: Compute Residuals using `hdm::rlasso()` ===
Y_resid <- rep(NA, N)
D_resid <- rep(NA, N)

for (k in 1:K) {
    # Training and validation split
    train_idx <- unlist(folds[-k])  # Use all but k-th fold for training
    val_idx <- folds[[k]]  # Use k-th fold for validation

    # Training Data
    X_train <- X[train_idx, ]
    Y_train <- Y[train_idx]
    D_train <- D[train_idx]

    # Validation Data
    X_val <- X[val_idx, ]

    # ---- Fit Lasso for Outcome Model (Y ~ X) using `hdm::rlasso()` ----
    lasso_y <- rlasso(X_train, Y_train)
    Y_hat_val <- predict(lasso_y, X_val)

    # ---- Fit Lasso for Treatment Model (D ~ X) using `hdm::rlasso()` ----
    lasso_d <- rlasso(X_train, D_train)
    D_hat_val <- predict(lasso_d, X_val)

    # ---- Compute Residuals ----
    Y_resid[val_idx] <- Y[val_idx] - Y_hat_val
    D_resid[val_idx] <- D[val_idx] - D_hat_val
}

# === Step 4: Estimate ATE Using HC3/EHW robust SE ===
dml2_model_robust <- lm_robust(Y_resid ~ D_resid, se_type = "HC3")

# === Step 5: Print Final Results ===
cat("\nEstimated ATE using DML2 (`hdm::rlasso`):", coef(dml2_model_robust)[2], "\n")
```

```
## 
## Estimated ATE using DML2 (`hdm::rlasso`): 1.99418
```

``` r
cat("Standard Error (lm_robust):", dml2_model_robust$std.error[2], "\n")
```

```
## Standard Error (lm_robust): 0.02366582
```

``` r
cat("95% CI (lm_robust):", dml2_model_robust$conf.low[2], "to",
    dml2_model_robust$conf.high[2], "\n")
```

```
## 95% CI (lm_robust): 1.94779 to 2.04057
```






### Sparsity in DML-Lasso:

Sparsity implies that only a few covariates in \(X\) are truly influential in predicting the outcome \(Y\) or the treatment variable \(D\). In Double Machine Learning (DML), the core assumption is one of "approximate sparsity," meaning that the true underlying functions \(f(X)\) and \(g(X)\) can be *well-approximated* by a sparse linear model. In Lasso \(f(X)\) represents the function that links the covariates \(X\) to the outcome \(Y\), while \(g(X)\) captures the relationship between \(X\) and the treatment variable \(D\), often reflecting the propensity score model. In practical terms, even if a dataset contains many covariates, only a limited number drive the key relationships, which is crucial for the Lasso method. Lasso effectively identifies these important predictors while discarding less relevant ones, thereby preventing the model from over-shrinking the influence of significant variables.

Footnote [^appspar] explains this concept mathematically by showing that the best sparse approximation of \(f(X)\) yields an error that diminishes as the number of effective covariates grows more slowly than the sample size \(N\).

[^appspar]: Mathematically, Approximate Sparsity Implies – The best sparse approximation of \(f(X)\) satisfies for simple Lasso models (in low-dimensional settings as described by Hastie, Tibshirani, and Friedman (2009)):\(\min_{\|\beta\|_0 \leq s} \mathbb{E} \left[ (f(X) - X' \beta)^2 \right] = O(s / N)\) where \(s\) is the effective number of relevant covariates and \(N\) is the sample size. This means that the error from using only the best \(s\) variables diminishes as \(s\) grows more slowly than \(N\). Lasso is effective if most of the explanatory power in \(f(X)\) and \(g(X)\) is concentrated in a small number of covariates.

When the sparsity condition holds, Lasso serves as a valid estimator for the nuisance functions \(\hat{Y}(X)\) and \(\hat{p}(X)\). This results in minimal bias from omitted variables and maintains the Neyman orthogonality condition, which ensures that small inaccuracies in the estimation of these nuisance functions do not lead to substantial bias in the final estimate of \(\hat{\delta}\). On the other hand, while Lasso is effective at identifying important predictors, it can sometimes shrink the coefficients of truly important variables too much, especially in the presence of highly correlated predictors. This can lead to underestimation of the true effects of these variables. Nonetheless, Lasso remains valuable in practice for its ability to perform variable selection.

This discussion underpins the use of three Lasso-based approaches: Double Lasso – Partialling-Out, Double Lasso – Double Selection, and Double Machine Learning (DML1 & DML2). The sparsity assumption is vital because it allows Lasso to approximate the nuisance functions accurately while keeping selection bias under control. Following Bühlmann and Van De Geer (2011) and Belloni, Chernozhukov, and Hansen (2014), the sparsity condition for the Double Lasso-Partialling-Out and Double Selection- methods is typically framed as requiring that \(\frac{s}{\sqrt{N}/\ln p}\) remains small, ensuring that the model correctly identifies important variables without overfitting.

In contrast, Chernozhukov et al. (2018) demonstrate that the use of cross-fitting in DML1 and DML2 relaxes the sparsity condition to \(\frac{s}{N/\ln p}\). Because \(N\) is much larger than \(\sqrt{N}\), cross-fitting permits a greater number of covariates while still yielding valid inference. For instance, with \(s = 4\), \(p = 10\), and \(N = 10,000\), the sparsity condition is approximately 0.092 for high-dimensional Lasso but drops to 0.000921 for cross-fitting DML, meaning that cross-fitting permits a larger number of covariates while maintaining valid inference. As a result, DML2 is often recommended when there is concern about meeting the sparsity requirement.


``` r
N <- 10000
p <- 10
s <- 4  # Example: Lasso selects 4 variables

sparsity_standard <- s / (sqrt(N) / log(p)) # Double Lasso - Partialling-Out
sparsity_crossfit <- s / (N / log(p)) # Double Machine Learning (DML1, DML2)

cat("Sparsity Condition (Standard):", sparsity_standard, "\n")
```

```
## Sparsity Condition (Standard): 0.0921034
```

``` r
cat("Sparsity Condition (Cross-Fitting):", sparsity_crossfit, "\n")
```

```
## Sparsity Condition (Cross-Fitting): 0.000921034
```


## DML2 -Decision Trees

In Chapter 14, we introduced decision trees as a flexible nonparametric approach for prediction. Here, we extend their use to causal inference by implementing a double machine learning (DML) approach with cross-fitting.

This simulation estimates the average treatment effect (ATE) using decision trees. The dataset consists of an outcome variable \( Y \), a treatment variable \( D \), and covariates \( X \). The treatment assignment follows a logistic model, and the outcome is a linear function of the treatment and covariates with added noise. We use decision trees to estimate two models: the outcome model \( Y \sim X \) and the treatment model \( D \sim X \). The treatment model is a classification tree predicting the probability of receiving treatment.

In the decision tree regression for \( Y \sim X \), the `rpart` package constructs a single tree that recursively partitions the feature space into smaller regions. Unlike linear models, which assume a specific functional form between \( Y \) and \( X \), decision trees adaptively split the data based on feature interactions and nonlinearities. The model chooses the optimal split at each node by minimizing the residual sum of squares (RSS), ensuring that the resulting partitions maximize within-node homogeneity.

By default, the `rpart` package in R uses complexity parameter (`cp = 0.01`) to prune unnecessary splits and prevent overfitting. Each split is only retained if it reduces RSS by at least 1%. Additionally, the minimum bucket size (`minbucket = 7` for regression) ensures that terminal nodes contain at least 7 observations, limiting excessive branching. The tree is grown deeply and then pruned back based on cost-complexity criteria. The final prediction for \( Y \) is obtained by assigning each observation to the mean outcome within its terminal node:

\begin{equation}
\hat{Y}(X) = \frac{1}{N_r} \sum_{i \in r} Y_i
\end{equation}

where \( N_r \) is the number of observations in region \( r \), and the sum is taken over all observations in that region.

For the treatment model \( D \sim X \), we apply a classification decision tree using the Gini impurity criterion to determine optimal splits. Unlike regression trees, which minimize RSS, classification trees split on the feature that maximally reduces impurity across child nodes. The treatment probability (propensity score) is then estimated by calculating the proportion of treated observations within each terminal node:

\begin{equation}
\hat{P}(D = 1 \mid X) = \frac{\sum_{i \in r} \mathbb{1}(D_i = 1)}{N_r}
\end{equation}

where \( \mathbb{1}(D_i = 1) \) is an indicator function for treated individuals in region \( r \), and \( N_r \) is the number of observations in that region.

To prevent excessive overfitting, the tree is constrained using `minbucket = 10` and `cp = 0.001`, ensuring that each leaf has at least 10 observations and that additional splits must improve model fit significantly. Unlike random forests, which smooth probability estimates by averaging over multiple trees, decision trees estimate treatment probabilities in a stepwise manner, leading to sharp discontinuities between regions.

To reduce overfitting, we apply cross-fitting by splitting the data into \( K \)-folds. In each fold, decision trees are trained on a subset of the data, and predictions are made on the held-out fold as we discussed above. The residuals from these models (i.e., differences between actual and predicted values) are then used in a final regression to estimate the treatment effect, controlling for confounding.

This method improves robustness by reducing overfitting and mitigating bias from direct regression of \( Y \) on \( D \). The final step uses heteroskedasticity-robust standard errors to provide valid inference. By using decision trees, this approach flexibly models nonlinear relationships but may suffer from high variance and sensitivity to small changes in data, especially compared to ensemble methods like random forests.


``` r
# Load required libraries
library(MASS)       # For generating multivariate normal data
library(caret)      # For cross-validation
library(estimatr)   # For HC3 robust SEs (lm_robust)
library(rpart)      # For decision trees
set.seed(123)       # For reproducibility

# === Step 1: Generate Simulated Data (Y, D, X) ===
N <- 10000  # Number of observations
p <- 10     # Number of covariates

# Generate covariates X ~ N(0, I)
X <- mvrnorm(N, mu = rep(0, p), Sigma = diag(p))

# Generate treatment variable D using a logistic model
D_prob <- plogis(X %*% runif(p, -1, 1))
D <- rbinom(N, 1, D_prob)

# Generate outcome Y with treatment effect
gamma_true <- 2
beta <- runif(p, -1, 1)
Y <- gamma_true * D + X %*% beta + rnorm(N)

# === Step 2: Define Fold Assignments (Cross-Fitting) ===
K <- 5 
folds <- createFolds(1:N, k = K, list = TRUE)

# === Step 3: Compute Residuals using decision trees (rpart) ===
Y_resid <- rep(NA, N)
D_resid <- rep(NA, N)

for (k in 1:K) {
  # Define training and validation indices
  train_idx <- unlist(folds[-k])
  val_idx <- folds[[k]]
  
  # Prepare validation data as a data frame with proper column names
  val_data <- as.data.frame(X[val_idx, ])
  
  # Outcome model: Fit a regression tree for Y ~ X
  train_data_y <- cbind(data.frame(Y = Y[train_idx]), as.data.frame(X[train_idx, ]))
  outcome_tree <- rpart(as.formula("Y ~ ."), data = train_data_y,
                        minbucket = 10, cp = 0.001)
  Y_hat_val <- predict(outcome_tree, newdata = val_data)
  
  # Treatment model: Fit a classification tree for D ~ X
  # Convert D to factor to use classification mode
  train_data_d <- cbind(data.frame(D = factor(D[train_idx])),
                        as.data.frame(X[train_idx, ]))
  treatment_tree <- rpart(as.formula("D ~ ."), data = train_data_d,
                          method = "class", minbucket = 10, cp = 0.001)
  # Obtain predicted probabilities for class "1"
  pred_probs <- predict(treatment_tree, newdata = val_data, type = "prob")
  D_hat_val <- pred_probs[, "1"]
  
  # Compute residuals
  Y_resid[val_idx] <- Y[val_idx] - Y_hat_val
  D_resid[val_idx] <- D[val_idx] - D_hat_val
}

# === Step 4: Estimate ATE Using HC3/EHW robust SE ===
dml2_model_robust <- lm_robust(Y_resid ~ D_resid, se_type = "HC3")

# === Step 5: Print Final Results ===
cat("\nEstimated ATE using DML2 (decision trees):", coef(dml2_model_robust)[2], "\n")
```

```
## 
## Estimated ATE using DML2 (decision trees): 1.620865
```

``` r
cat("Standard Error (lm_robust):", dml2_model_robust$std.error[2], "\n")
```

```
## Standard Error (lm_robust): 0.02828379
```

``` r
cat("95% CI (lm_robust):", dml2_model_robust$conf.low[2], "to",
    dml2_model_robust$conf.high[2], "\n")
```

```
## 95% CI (lm_robust): 1.565423 to 1.676307
```

Decision trees likely gave the worst results in the DML framework due to several well-known limitations of single decision trees, especially in high-dimensional or noisy data settings:

1. **High Variance and Instability**:  
   - Decision trees tend to overfit the training data because they create very specific splits based on the given sample.  
   - Unlike ensemble methods (e.g., random forests), a single tree lacks regularization, making it sensitive to small changes in the training data.  
   - This instability can lead to large errors in the residualized treatment (\(\hat{D} - D\)) and outcome (\(\hat{Y} - Y\)) models, which propagate to the final treatment effect estimate.

2. **Limited Predictive Power for Complex Relationships**:  
   - Decision trees partition the data into discrete regions, making them inefficient at capturing smooth, continuous relationships between variables.  
   - If \( Y \) and \( D \) are influenced by interactions or nonlinear effects that are not well captured by splits, the residualization step will be poor, leading to biased or inconsistent treatment effect estimates.

3. **Overfitting to the Training Data**:  
   - Trees tend to perfectly fit training data when grown deep, but this does not generalize well to new samples.  
   - This means the estimated functions \( \hat{Y}(X) \) and \( \hat{D}(X) \) may be highly inaccurate when evaluated on held-out folds, worsening cross-fitting results.

4. **Poor Handling of Continuous Treatment Probabilities**:  
   - In the treatment model \( D \sim X \), a classification tree is used to predict propensity scores \( P(D=1 \mid X) \).  
   - Unlike logistic regression or random forests, decision trees only create stepwise splits, which can result in poorly estimated probabilities.  
   - This leads to high residual variance, reducing the effectiveness of the second-stage regression.

Random forests mitigate many of these issues by averaging multiple decision trees trained on bootstrapped samples and random feature subsets. This reduces variance, improves generalization, and provides smoother propensity score estimates for treatment assignment. Since DML relies on accurate nuisance function estimates (\(\hat{Y}(X)\) and \(\hat{D}(X)\)), a poorly performing base model (like a single decision tree) leads to biased or highly variable ATE estimates.

In summary, decision trees struggle due to overfitting, poor smoothness, and high variance, which weakens the residualization step in DML, making them less reliable for causal inference.

## DML2 - Random Forest

In Chapter 15, we covered random forests for prediction. Here, we extend their use to causal inference by implementing a DML approach with cross-fitting.

This simulation estimates the ATE using random forests. The dataset consists of an outcome variable \( Y \), a treatment variable \( D \), and covariates \( X \). The treatment assignment follows a logistic model, and the outcome is a linear function of the treatment and covariates with added noise. We use random forests to estimate two models: the outcome model \( Y \sim X \) and the treatment model \( D \sim X \). The treatment model is a regression forest predicting the probability of receiving treatment.

In the random forest regression for \( Y \sim X \), the `randomForest` package constructs an ensemble of decision trees to predict \( Y \) based on the covariates \( X \). Unlike a single decision tree, which partitions the feature space into non-overlapping regions, a random forest builds multiple trees on different bootstrap samples of the data and averages their predictions. This process helps reduce overfitting and improves generalization. Each individual tree is grown using a subset of randomly selected features at each split rather than considering all features, which introduces decorrelation between trees and further improves robustness.

By default, the `randomForest` package in R sets the number of trees to 500 (`ntree = 500`) and selects \(\lfloor \sqrt{p} \rfloor\) features for each split when performing regression (`mtry = floor(sqrt(p))`). The trees are grown deep, with splits determined by minimizing the residual sum of squares (RSS) at each node until a stopping criterion is met. The default minimum node size (`nodesize = 5` for regression) controls the depth of the trees, preventing overfitting by requiring at least 5 observations in a terminal node before stopping further splits.

Increasing the number of trees (`ntree`) generally reduces variance, leading to more stable predictions. While 500 trees is often sufficient, increasing this to 1000 or more can help in high-dimensional settings or when working with noisy data. However, returns diminish beyond a certain point, as additional trees provide only marginal improvements in performance but increase computational cost. A practical approach is to start with 500 trees, monitor the out-of-bag (OOB) error, and increase `ntree` until the error stabilizes.

The minimum node size (`nodesize`) controls the granularity of splits and influences the bias-variance tradeoff. The default value of 5 results in deeper trees that capture fine-grained patterns but may overfit. Setting `nodesize = 10` or higher prunes the trees earlier, reducing overfitting by forcing more generalization. If the data is noisy, increasing `nodesize` (e.g., 10-20) can improve robustness by smoothing predictions. A common rule of thumb is to set `nodesize` to approximately 1% of total observations, though the optimal choice often depends on cross-validation. In causal inference, larger node sizes are preferred to avoid overfitting the treatment and outcome models, ensuring more reliable effect estimates.

Once all trees are grown, the final prediction for \( Y \) is obtained by averaging the predictions from all trees: 

\begin{equation}
\hat{Y}(X) = \frac{1}{B} \sum_{b=1}^{B} \hat{f}^{(b)}(X)
\end{equation}

where \( B \) is the number of trees, and \( \hat{f}^{(b)}(X) \) is the prediction from the \( b \)-th tree.

The same procedure is applied to estimate \( D \sim X \), but since \( D \) is binary (treatment assignment), classification random forest is used instead. The splitting criterion in classification trees is different—by default, it minimizes the Gini impurity rather than RSS. Instead of averaging numerical predictions, classification forests output class probabilities, with the predicted probability for treatment (\( D = 1 \)) given by:  

\begin{equation}
\hat{P}(D = 1 \mid X) = \frac{1}{B} \sum_{b=1}^{B} \hat{P}^{(b)}(D = 1 \mid X)
\end{equation}

where each tree \( \hat{P}^{(b)} \) provides a probability estimate based on the proportion of \( D = 1 \) observations in each terminal node. This probability serves as the estimated propensity score, which is used in the residualization step for double machine learning.

To reduce overfitting, we apply cross-fitting by splitting the data into \( K \)-folds. In each fold, random forests are trained on a subset of the data, and predictions are made on the held-out fold as we discussed above. The residuals from these models (i.e., differences between actual and predicted values) are then used in a final regression to estimate the treatment effect, controlling for confounding.

This method improves robustness by reducing overfitting and mitigating bias from direct regression of \( Y \) on \( D \). The final step uses heteroskedasticity-robust standard errors to provide valid inference. By using random forests, this approach flexibly captures complex nonlinear relationships while ensuring that selection bias from observed covariates is accounted for.


``` r
# Load required libraries
library(MASS)       # For generating multivariate normal data
library(caret)      # For cross-validation
library(estimatr)   # For HC3 robust SEs (lm_robust)
library(randomForest)  # For random forest modeling
set.seed(123)       # Set seed for reproducibility

# === Step 1: Generate Simulated Data (Y, D, X) ===
N <- 10000  # Number of observations
p <- 10     # Number of covariates

# Generate covariates X ~ N(0, I)
X <- mvrnorm(N, mu = rep(0, p), Sigma = diag(p))

# Generate treatment variable D using a logistic model
D_prob <- plogis(X %*% runif(p, -1, 1))  # Compute propensity scores
D <- rbinom(N, 1, D_prob)                # Assign treatment based on propensity

# Generate outcome Y with treatment effect
gamma_true <- 2                        # True treatment effect
beta <- runif(p, -1, 1)                  # True coefficients for X
Y <- gamma_true * D + X %*% beta + rnorm(N)  # Outcome model

# === Step 2: Define Fold Assignments (Cross-Fitting) ===
K <- 5 
folds <- createFolds(1:N, k = K, list = TRUE)

# === Step 3: Compute Residuals using random forest ===
Y_resid <- rep(NA, N)
D_resid <- rep(NA, N)

for (k in 1:K) {
  # Define training and validation indices
  train_idx <- unlist(folds[-k])
  val_idx <- folds[[k]]
  
  # Prepare data for outcome model
  train_data_y <- data.frame(Y = Y[train_idx], as.data.frame(X[train_idx, ]))
  val_data <- as.data.frame(X[val_idx, ])
  
  # Fit random forest for outcome (Y ~ X)
  rf_y <- randomForest(Y ~ ., data = train_data_y, ntree = 1000, nodesize = 10)
  Y_hat_val <- predict(rf_y, newdata = val_data)
  
  # Prepare data for treatment model
  train_data_d <- data.frame(D = D[train_idx], as.data.frame(X[train_idx, ]))
  
  # Fit random forest for treatment (D ~ X)
  rf_d <- randomForest(D ~ ., data = train_data_d, ntree = 1000, nodesize = 10)
  D_hat_val <- predict(rf_d, newdata = val_data)
  
  # Compute residuals
  Y_resid[val_idx] <- Y[val_idx] - Y_hat_val
  D_resid[val_idx] <- D[val_idx] - D_hat_val
}

# === Step 4: Estimate ATE Using HC3/EHW robust SE ===
dml2_model_robust <- lm_robust(Y_resid ~ D_resid, se_type = "HC3")

# === Step 5: Print Final Results ===
cat("\nEstimated ATE using DML2 (random forest):", coef(dml2_model_robust)[2], "\n")
```

```
## 
## Estimated ATE using DML2 (random forest): 1.950607
```

``` r
cat("Standard Error (lm_robust):", dml2_model_robust$std.error[2], "\n")
```

```
## Standard Error (lm_robust): 0.0246481
```

``` r
cat("95% CI (lm_robust):", dml2_model_robust$conf.low[2], "to", 
    dml2_model_robust$conf.high[2], "\n")
```

```
## 95% CI (lm_robust): 1.902292 to 1.998922
```
    
## Technical Notes: From FWL to DML
### Frisch-Waugh-Lovell Theorem 

The Frisch-Waugh-Lovell (FWL) theorem explains why ordinary least squares (OLS) regression adjustment effectively isolates the treatment effect by controlling for confounders. It shows that in a multiple regression, you can isolate the impact of one variable by removing the effects of the other variables from both the dependent variable and the variable of interest. This process—known as partialing out—consists of regressing the variable of interest on the other predictors, then using those residuals in a second regression with the dependent variable. This yields the same coefficient as a full multiple regression. In essence, orthogonalization—the process of transforming variables so that they do not share any common variation with the other predictors—allows you to clearly see the unique contribution of each variable in the model.

For instance, in

\begin{equation}
y = X_1\beta_1 + X_2\beta_2 + \varepsilon
\end{equation}

the FWL theorem says \(\beta_1\) can be obtained by first regressing both \(y\) and \(X_1\) on \(X_2\) to get residuals orthogonal to \(X_2\). If

\begin{equation}
M_{X_2} = I - X_2(X_2^\top X_2)^{-1}X_2^\top
\end{equation}

is the residual-maker matrix, then

\begin{equation}
\tilde{y} = M_{X_2}y,\quad \tilde{X}_1 = M_{X_2}X_1
\end{equation}

Regressing \(\tilde{y}\) on \(\tilde{X}_1\) alone provides the same \(\beta_1\) as the full model. In double machine learning, a similar approach is used to create an orthogonal score by residualizing both the outcome and the treatment with respect to controls, thereby removing confounder effects.

Turning to causal or treatment effects, consider the OLS model:

\begin{equation}
Y_i = \alpha + \delta D_i + X_i^\top \beta + \varepsilon_i
\end{equation}

where \(Y_i\) is the outcome, \(D_i\) is the treatment, \(X_i\) represents observed confounders, and \(\varepsilon_i\) captures unobserved factors. Under

\begin{equation}
E[\varepsilon_i \mid D_i, X_i] = E[\varepsilon_i \mid X_i]
\end{equation}

OLS provides a consistent estimate of the average treatment effect (ATE), \(\delta\). The FWL theorem offers an equivalent way to compute \(\delta\) by residualizing both \(Y_i\) and \(D_i\) on \(X_i\), then regressing the outcome residuals on the treatment residuals. This makes clear that once \(X_i\) is accounted for, any remaining variation in \(D_i\) should be independent of unobserved determinants of \(Y_i\), mimicking a randomized experiment conditional on \(X_i\).

However, if there are unobserved confounders correlated with both treatment and outcome, simple regression adjustment cannot eliminate all bias, making methods like instrumental variables or difference-in-differences necessary. Also note that both FWL and OLS rely on standard assumptions (linearity, exogeneity, etc.): violations can lead to biased or inefficient estimates.

As we discussed before, in the direct OLS approach, under the assumption \(E[\varepsilon_i \mid D_i, X_i] = E[\varepsilon_i \mid X_i]\), OLS provides an unbiased estimate of \(\delta\). Standard practice is to use robust (Eicker-Huber-White) standard errors to account for potential heteroskedasticity.

In the FWL two-step approach, we first residualize \(Y_i\) and \(D_i\) with respect to \(X_i\), then regress these residuals. The new error term reflects only variation in \(Y_i\) not explained by \(X_i\). Because \(D_i\) is also residualized against \(X_i\), the new error term (from residualized regression) is uncorrelated with the residualized treatment, and robust standard errors remain valid. Although the computation differs, both methods yield the same estimate of \(\delta\) and identical statistical inference.

The following R simulation shows that both the full OLS model and the FWL approach yield identical treatment coefficients and robust standard errors. 

``` r
library(MASS)      # for mvrnorm
library(sandwich)  # for robust covariance estimates
set.seed(123)
N <- 10000; p <- 10
X <- mvrnorm(N, rep(0, p), diag(p))
D_prob <- plogis(X %*% runif(p, -1, 1))
D <- rbinom(N, 1, D_prob)
gamma_true <- 2
beta <- runif(p, -1, 1)
Y <- gamma_true * D + X %*% beta + rnorm(N)

# Full OLS model
full_model <- lm(Y ~ D + X)
cat("Full OLS coefficient (D):", coef(full_model)["D"], "\n")
```

```
## Full OLS coefficient (D): 1.996462
```

``` r
cat("Robust SE:", sqrt(diag(vcovHC(full_model, type = "HC1")))["D"], "\n\n")
```

```
## Robust SE: 0.02366501
```

``` r
# FWL approach:
# 1. Residualize Y and D on X
res_Y <- resid(lm(Y ~ X))
res_D <- resid(lm(D ~ X))
# 2. Regress residualized Y on residualized D
fwl_model <- lm(res_Y ~ res_D - 1)
cat("FWL coefficient (D):", coef(fwl_model)["res_D"], "\n")
```

```
## FWL coefficient (D): 1.996462
```

``` r
cat("Robust SE:", sqrt(diag(vcovHC(fwl_model, type = "HC1")))["res_D"], "\n")
```

```
## Robust SE: 0.02365199
```


### From FWL to Double Machine Learning (DML):

In this section, we bridge the classical Frisch–Waugh–Lovell (FWL) theorem with Double Machine Learning (DML) to illustrate how robust causal inference is achieved through orthogonalization. We first review the FWL approach—where the outcome \(Y\)and treatment \(D\) are residualized with respect to covariates \(X\) —and then show how these ideas extend naturally to modern DML methods using flexible machine learning tools such as LASSO. This unified framework underscores the importance of partialling out nuisance variation to isolate the causal effect of interest.

We adopt the notation from Chernozhukov et al. (2018) for our linear model:

\begin{equation}
Y = D\,\theta + X\,\beta + \varepsilon
\end{equation}

where the goal is to estimate \(\theta\), the coefficient on \(D\). According to the Frisch–Waugh–Lovell theorem, we can “partial out” the effects of \(X\) by first regressing both \(Y\) and \(D\) on \(X\) and then using their residuals. Specifically:

1. **Regress \(Y\) on \(X\)** and let \(\hat{g}(X)\) be the fitted values. The residual becomes 

   \[
   \tilde{Y} = Y - \hat{g}(X)
   \]

2. **Regress \(D\) on \(X\)** and let \(\hat{m}(X)\) be the fitted values. The residual is 

   \[
   \tilde{D} = D - \hat{m}(X)
   \]

3. **Run OLS of \(\tilde{Y}\) on \(\tilde{D}\)**: According to FWL, the regression of \(Y\) on \(D\) controlling for \(X\) is equivalent to regressing \(\tilde{Y}\) on \(\tilde{D}\). This gives

   \[
   \tilde{Y} = \theta\,\tilde{D} + \tilde{\varepsilon}
   \]
   
To obtain \(\theta\) by OLS, we set up the first-order condition (the moment condition) by differentiating the sum of squared residuals with respect to \(\theta\). That condition is

\[
\frac{\partial}{\partial \theta} \sum_{i} \left(\tilde{Y}_i - \theta \tilde{D}_i\right)^2 = 0
\]

Differentiating ( and Dividing by \(-2\)), we have

\[
\sum_{i} \tilde{D}_i\left(\tilde{Y}_i - \theta \tilde{D}_i\right) = 0
\]

In expectation (or in sample average form), we write the moment condition as

\[
E\Big[\tilde{D}\big(\tilde{Y} - \theta \tilde{D}\big)\Big] = 0
\]

Expanding this condition gives

\[
E[\tilde{D}\tilde{Y}] - \theta E[\tilde{D}^2] = 0
\]

Solving for \(\theta\), we have

\[
\theta = \frac{E[\tilde{D}\tilde{Y}]}{E[\tilde{D}^2]}
\]

In a sample, this matches the OLS estimate from regressing \(\tilde{Y}\) on \(\tilde{D}\). This derivation shows step by step how the coefficient on \(D\) is obtained by solving the moment condition arising from the residualized variables, and \(\theta\) reflects the causal/treatment effect of \(D\) net of the controls.

This derivation also reveals the orthogonal score in the context of OLS. Defining

\[
\psi(W; \theta) = \tilde{D}\,(\tilde{Y} - \theta\,\tilde{D})
\]

we see that once \(Y\) and \(D\) are residualized on \(X\), the score depends only on the variation of \(Y\) and \(D\) that is uncorrelated with \(X\). In other words, the orthogonality condition is given by

\[
E[\psi(W; \theta)]=E\big[\tilde{D}(\tilde{Y} - \theta\,\tilde{D})\big] = 0
\] 

This idea underlies the orthogonal score used in double machine learning, protecting the \(\theta\) estimate from small errors in the nuisance function(s) \(\hat{g}(X)\) and \(\hat{m}(X)\) (here \(\eta=1\) since we use a single set of nuisance functions, as opposed to using cross-fitted residuals in DML).

Notably, the coefficient \(\theta\) in the final expression, and orthogonal score above coincides with the “debiased” or “partialled-out” estimator and orthogonal score function discussed in Chernozhukov et al. (2018), particularly in Sections 4 (and footnote 3 therein). The partialing-out procedure and the DML1 and DML2 algorithms’ score function is exactly the same as their equation 4.55 in Section 4-partially linear models, just framed here in a simpler linear setting.

Having established that the FWL theorem yields an orthogonal score and unbiased treatment effect estimates by residualizing \(Y\) and \(D\), we now turn to Double Machine Learning. In this approach, we incorporate advanced ML techniques to estimate the nuisance functions, while preserving the orthogonality property that safeguards our causal estimates even in high-dimensional settings.

**Extending to Double Machine Learning**

Double or Debiased Machine Learning (DML) extends the classical FWL theorem to settings where flexible machine learning methods can be used. The main idea is to estimate the outcome \(Y\) and the treatment \(D\) separately using ML techniques, then obtain residuals by partialling out the contribution of \(X\), and finally regress the outcome residual on the treatment residual. Formally, one first estimates

\begin{equation}
\hat{g}(X) \approx E[Y \mid X] \quad \text{and} \quad \hat{m}(X) \approx E[D \mid X]
\end{equation}

using methods such as LASSO. One then forms the residuals

\begin{equation}
\tilde{Y} = Y - \hat{g}(X) \quad \text{and} \quad \tilde{D} = D - \hat{m}(X)
\end{equation}

and regresses \(\tilde{Y}\) on \(\tilde{D}\) to obtain the treatment-effect parameter \(\theta\). This procedure generalizes the standard partialling-out approach to high-dimensional or complex data scenarios while preserving a key principle: orthogonality.

Orthogonality arises because the residuals from the outcome and treatment regressions remove most of the variation associated with \(X\). As a result, the estimator of the average treatment effect (ATE) is robust to moderate errors in estimating these nuisance functions. In a typical treatment-effects context, the goal is to identify a parameter \(\theta\) in the partially linear model

\[
Y = \theta\,D + f(X) + \varepsilon
\]

where \(f(X)\) is a nuisance function (which is any part of the model not containing the main parameter of interest). Unbiased estimation of \(\theta\) requires proper accounting for \(f(X)\). In DML, one estimates the nuisance functions \(\hat{g}(X)\) for the outcome and \(\hat{m}(X)\) for the treatment, and then constructs an orthogonal score of the form

\begin{equation}
\psi(W; \theta, \eta) = \bigl(Y - \hat{g}(X)\bigr)\,\bigl(D - \hat{m}(X)\bigr) - \theta\,\bigl(D - \hat{m}(X)\bigr)^2
\end{equation}

where \(W = (Y, D, X)\) and \(\eta\) denotes the collection of nuisance functions (this equation can be shortened as \((\tilde{Y} - \theta\,\tilde{D})\tilde{D}\), which is the same as the equation obtained above using FWL, and is exactly the same as equation 4.55 in Chernozhukov et al. (2018)). The final estimate of \(\theta\) is obtained by solving the following moment condition for \(\theta\) (similar to the final OLS step) of

\begin{equation}
E[\psi(W; \theta, \eta)] = 0
\end{equation}

Because the derivative of this moment condition with respect to the nuisance functions is zero at the true parameter, small errors in estimating \(\hat{g}(X)\) or \(\hat{m}(X)\) do not substantially bias \(\theta\).[^DerOtrh] This orthogonality property is a key feature of DML, ensuring that the treatment effect estimate is robust to moderate errors in the nuisance functions.

[^DerOtrh]:Derivation of the Orthogonality Condition via the Gateaux Derivative: Consider the moment function
\[
\psi(W; \theta, \eta) = \bigl(D - m(X)\bigr)\Bigl(Y - g(X) - \theta\,(D - m(X))\Bigr)
\]
with \(\eta = (g, m)\) and the true nuisance functions \(\eta_0 = (g_0, m_0)\), where \(g_0(X) = E[Y\mid X]\) and \(m_0(X) = E[D\mid X]\). The population moment condition at the true parameter \(\theta_0\) is
\[
E\bigl[\psi(W; \theta_0, \eta_0)\bigr] = 0
\]
To study the sensitivity of this moment condition to small perturbations in the nuisance functions, we consider a directional perturbation \(h = (h_g, h_m)\) and define the Gateaux derivative as
\[
\frac{\partial}{\partial \eta}\,E\bigl[\psi(W; \theta_0, \eta)\bigr]\Big|_{\eta = \eta_0}[h] = \lim_{t\to 0} \frac{E\bigl[\psi(W; \theta_0, \eta_0 + t\,h)\bigr] - E\bigl[\psi(W; \theta_0, \eta_0)\bigr]}{t}
\]
Since \(E[\psi(W; \theta_0, \eta_0)] = 0\), the orthogonality condition (or Neyman orthogonality) requires that
\[
\frac{\partial}{\partial \eta}\,E\bigl[\psi(W; \theta_0, \eta)\bigr]\Big|_{\eta = \eta_0}[h] = 0
\]
for all admissible directions \(h\). This means that first-order changes in the nuisance functions (i.e., small estimation errors) do not affect the moment condition—any impact is only of second order. This insensitivity is central to double machine learning, as it protects the estimate of \(\theta\) from moderate errors in estimating the nuisance functions \(g_0(X)\) and \(m_0(X)\).

A key element of DML is controlling overfitting, which can bias the residuals if the ML procedure (in estimating nuisance functions) captures too much of the true relationship between \(X\) and \(Y\) or \(D\). Overfitting in \(\hat{g}(X)\) may inadvertently capture variation in \(Y\) that is due to \(D\), leading to downward-biased \(\tilde{Y}\), while overfitting in \(\hat{m}(X)\) can reduce the variance in \(\tilde{D}\) and increase the variance of the final estimator (\(\theta\) here). To mitigate these issues, DML employs sample splitting (or cross-fitting), partitioning the data into folds. In another words, the ML models are trained on all but one fold, and the held-out fold is used to generate out-of-sample predictions (to calculate residuals) to prevent bias for the nuisance functions. Repeating and combining these predictions systematically across all folds to estimate the final model correct overfitting bias.

Naive machine learning approaches that fail to account for orthogonality (i.e., directly implementing LASSO or random forest in a partially linear model instead of residualizing \(Y\) and \(D\)) may yield biased estimates because moderate errors in the nuisance functions can cause large errors in the final parameter of interest. DML’s orthogonalization explicitly addresses this issue. By removing the main signal of \(X\) from both \(Y\) and \(D\), the procedure yields residuals that are, loosely speaking, “purified” of spurious variation and captures mainly the variation attributable to the relationship between \(Y\) and \(D\). 

In DML, bias from overfitting is mitigated via cross-fitting and orthogonality condition, while the use of LASSO in the training stage may also introduce shrinkage bias. LASSO (or similar regularized linear methods) is a natural choice in DML because it effectively handles high-dimensional settings by regularizing and selecting the most relevant covariates. For example, one can estimate \(\hat{g}(X)\) by regressing \(Y\) on \(X\) using LASSO and \(\hat{m}(X)\) by regressing \(D\) on \(X\) (employing logistic LASSO when \(D\) is binary). LASSO penalizes large coefficients, thereby controlling model complexity but also biasing estimates toward zero. This shrinkage bias is mitigated in practice by using cross-validation (or a similar tuning procedure) to select the penalty parameter \(\lambda\), which balances bias and variance to minimize out-of-sample prediction error during the training of the nuisance functions. To emphasize, LASSO in training stage of DML may still exhibit finite-sample shrinkage bias for inference, the orthogonality property in DML ensures that the final estimate of \(\theta\) remains unbiased even if the nuisance function estimates are moderately imprecise.[^posource]

[^posource]:Methods like post-LASSO (refitting the model on variables selected by LASSO; default in `hrm` package `rlasso`) or debiased LASSO (last section) provide additional corrections for valid hypothesis testing and confidence intervals, thereby mitigating the residual shrinkage bias in coefficient estimates directly.

**DML1 (Definition 3.1)**

Below is a concise restatement of **DML1** and **DML2** (Definitions 3.1 and 3.2 in Chernozhukov et al. 2018) that complements the discussion above. We focus on the key algorithmic steps and how they fit into the framework of orthogonalized estimating equations. For detailed implementation, please refer back to the practical instructions and code in the main text.

1. **Partition the Data**  
   Randomly split the sample into \(K\) folds \(\{I_k\}_{k=1}^K\), each containing \(n/K\) observations (assuming \(n\) is divisible by \(K\) for simplicity).

2. **Estimate Nuisance Functions** and **Form Cross-Fitted Residuals** 
   For each fold \(k\in \{1,\dots,K\}\):  
   - **Train** machine learning estimators \(\hat{\eta}_{-k}\) (e.g., \(\hat{g}_{-k}(X)\) and \(\hat{m}_{-k}(X)\)) on all folds except \(k\).  
   - **Predict** on fold \(k\), producing out-of-sample estimates for that fold.

3. **Construct the Orthogonal Moment for Each Fold**  
   Using the held-out fold \(k\), form the score function \(\psi\bigl(W_i; \theta, \hat{\eta}_{-k}\bigr)\) (see the previous text for the explicit formula). The estimator \(\hat{\theta}_k\) on fold \(k\) solves the empirical moment equation  
   
   \[
   \frac{1}{\lvert I_k\rvert} \sum_{i \in I_k} \psi\bigl(W_i; \theta, \hat{\eta}_{-k}\bigr) = 0
   \]
   
   which corresponds to Equation (3.38) in Chernozhukov et al. (2018).

4. **Combine Fold-Specific Estimates**  
   Average the solutions \(\hat{\theta}_k\) across all folds to obtain the final DML1 estimator  
   
   \[
   \hat{\theta}_{\mathrm{DML1}} = \frac{1}{K} \sum_{k=1}^K \hat{\theta}_k
   \]

This fold-by-fold approach produces one estimate per fold and then aggregates them. The moment equation is solved separately within each hold-out set, ensuring that nuisance function estimation and treatment-effect estimation use disjoint subsets of data.

**DML2 (Definition 3.2)**

1. **Partition and Estimate Nuisance Functions**  
   As in DML1, randomly split the data into \(K\) folds and for each fold \(k\), train \(\hat{\eta}_{-k}\) on all but the \(k\)-th fold, then predict on fold \(k\).

2. **Estimate Nuisance Functions**  and **Form Cross-Fitted Residuals**  
   Using \(\hat{\eta}_{-k}\), construct residuals \(\tilde{Y}_i\) and \(\tilde{D}_i\) for all \(i \in I_k\). Gather these residuals for all folds into a single sample.

3. **Solve a Single Moment Condition**  
   Unlike DML1, DML2 combines all folds’ residuals at once and solves a single orthogonal moment equation over the entire sample (Equation (3.41) in Chernozhukov et al. (2018)):  
   \[
   \frac{1}{n} \sum_{k=1}^K \sum_{i \in I_k}
   \psi\bigl(W_i; \theta, \hat{\eta}_{-k}\bigr) = 0
   \]
   yielding a single \(\hat{\theta}_{\mathrm{DML2}}\).

4. **Approximate Solutions**  
   If an exact solution is not possible, an approximate \(\epsilon\)-solution is employed (Equation (3.42) in Chernozhukov et al.), ensuring \(\hat{\theta}_{\mathrm{DML2}}\) is still valid under mild conditions.

#### Connection to Orthogonality and FWL

Both DML1 and DML2 rely on the same orthogonal score construction discussed above: the partialling out of \(X\) from \(Y\) and \(D\), followed by a moment condition that is insensitive (to first order) to errors in the nuisance functions. DML1 solves the orthogonal equation fold by fold, then averages, while DML2 pools all cross-fitted residuals and solves one global equation. In either case, the fundamental principle is that once \(Y\) and \(D\) are “purified” of the variation explained by \(X\), the remaining variation identifies the causal parameter \(\theta\). By using cross-fitting, both procedures mitigate overfitting bias and maintain valid inference in high-dimensional or complex data settings. Because of orthogonality, small inaccuracies in the nuisance regressions have a limited effect on the final parameter estimates, offering a powerful tool for causal inference when there are many covariates or complex relationships between regressors and outcomes.


### Debiased LASSO

Standard LASSO estimates are biased due to the \(\ell_1\)-penalty shrinking coefficients toward zero. Debiased LASSO (also known as desparsified or de-sparsified LASSO) corrects this shrinkage by adding a carefully constructed adjustment term. Pioneered by Zhang and Zhang (2014), van de Geer et al. (2014), and Javanmard and Montanari (2014), the procedure starts with the LASSO estimate \(\hat{\beta}_{\text{Lasso}}\), then updates it via

\begin{equation}
\hat{\beta}_{dl} 
= 
\hat{\beta}_{\text{Lasso}}
+
\frac{1}{n}\,\hat{\Theta}\,X^\top\bigl(Y - X\,\hat{\beta}_{\text{Lasso}}\bigr)
\end{equation}

Here, \(\hat{\Theta}\) approximates \(\Sigma^{-1}\), with \(\Sigma = \frac{1}{n}X^\top X\). In high-dimensional settings, \(\hat{\Theta}\) is often estimated by regressing each column of \(X\) on the other columns (nodewise regression) or other high-dimensional precision-matrix estimators. This one-step correction removes much of the bias introduced by LASSO’s penalty, allowing asymptotically valid confidence intervals for low-dimensional parameters (such as a specific treatment coefficient) even when \(p\) is large.


