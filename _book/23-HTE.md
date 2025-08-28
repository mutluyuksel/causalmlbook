# Heterogeneous Treatment Effects

\begin{itshape}
Why didn’t the Heterogeneous Treatment Effect (HTE) follow the standard treatment plan?
Because it knew one size doesn’t fit all!
\end{itshape}

In previous chapters, we primarily focused on the Average Treatment Effect (ATE) as a measure of causal impact across an entire population. However, ATE alone is not sufficient when treatment effects vary across individuals or groups. A policy based solely on ATE assumes uniform effects, which can lead to inefficiencies or unintended harm—some subgroups may benefit significantly, while others may see no effect or even experience adverse consequences. To address this, we introduced the Conditional Average Treatment Effect (CATE) in Chapter 18, which accounts for treatment heterogeneity by conditioning on observable characteristics. Estimating CATE allows for more precise policy targeting and personalized treatment assignment, improving decision-making in applied settings. With increasingly rich datasets and advanced machine learning tools, the estimation of treatment heterogeneity has gained widespread interest. In this chapter, we expand our discussion beyond individual-level heterogeneity to treatment effects at the group level and other measures of subgroup effects. A key objective is to systematically identify which subgroups benefit the most (or least) from an intervention. Machine learning can uncover this heterogeneity by modeling treatment effects as \(\delta^{CATE}(X) = \mathbb{E}[Y(1)-Y(0)\mid X]\), helping researchers and policymakers optimize interventions in a data-driven manner.

## Parametric Approaches to Estimating Heterogeneous Treatment Effects  

Parametric approaches represent a classical starting point for analyzing heterogeneous treatment effects.These methods assume a specific functional form for how the treatment effect varies with observed covariates across individuals or groups. These approaches provide interpretable and structured ways to estimate treatment heterogeneity while leveraging existing statistical techniques. A common strategy is to include interaction terms between the treatment indicator and covariates in aregression model. Other parametric methods involve stratified analysis, generalized linear models (GLMs), or random coefficients models. We discuss each of these approaches in turn. 

### Interaction Models in Linear Regression  

A simple way to capture treatment effect heterogeneity is by including interaction terms between the treatment indicator and covariates in a linear regression. In a randomized trial or observational study, one can estimate models like:  

\begin{equation}
Y_i = \alpha + \delta D_i + \beta X_i + \tau(D_i \times X_i) + \epsilon_i
\end{equation}

where \(D_i\) is the treatment indicator, \(X_i\) is a covariate (or vector of covariates), and \(\tau\) is the coefficient on the interaction term. The **Conditional Average Treatment Effect (CATE)** in this model is given by:

\begin{equation}
\tau(X) = \frac{\partial \mathbb{E}[Y \mid D, X]}{\partial D} = \delta + \tau X
\end{equation}

If \( \tau \neq 0 \), then the treatment effect depends on \( X \), indicating heterogeneity in responses, even at the individual level.  

This framework highlights that heterogeneity can be captured not only through predefined subgroups but also at the individual level when \(X\) is a continuous or categorical variable. When \(X\) is categorical, it defines subgroups, making subgroup analysis a special case of this interaction model. Instead of estimating separate regressions for different groups, we can use a single regression with group-treatment interactions, ensuring more efficient estimation while maintaining interpretability.  

Variables influencing outcomes exist at multiple levels. At the individual level, demographics (age, gender, race, education, income), behavioral factors (risk preferences, prior treatment, medication adherence), and health conditions (chronic disease, BMI, cognitive ability) matter. At the household level, socioeconomic status (income, parental education, dependents) and structure (single-parent household, siblings) play a role. The community level includes location-based factors (urban/rural, healthcare access, infrastructure) and policy exposure (state laws, minimum wage, economic conditions). At the institutional level, workplace characteristics (firm size, industry, unions) and school factors (class size, teacher experience, funding) shape outcomes. By incorporating these factors in an interaction model, we capture how treatment effects vary across these dimensions while maintaining a unified framework within a single regression.  

Interaction models are easy to implement (using OLS or GLM), and the coefficient \(\tau\) directly measures how the treatment effect changes per unit change in \(X\). However, with many potential covariates, testing all interactions can lead to multiple-comparison problems and overfitting, so usually, a few key moderators are pre-specified to avoid spurious findings.  

### Stratified or Subgroup Analysis  

So far, we have discussed parametric methods for estimating heterogeneous treatment effects, primarily through interaction models. A natural extension of this approach is the Group Average Treatment Effect (GATE), which measures treatment effects for predefined or data-driven subgroups. For subgroup analysis, we can estimate treatment effects using a single regression model with group-treatment interactions or, alternatively, separate regressions for predefined groups to capture systematic heterogeneity. We will discuss parametric models in this section, while the next sections will cover nonparametric approaches for more flexible estimation of treatment heterogeneity.  

#### **Single Regression**  

Often, policymakers and researchers are interested in the average treatment effect for a particular subgroup rather than a fully individualized effect. This is known as the GATE. Formally, if we define a subgroup (or grouping variable) \(G_i = g\) for unit \(i\) (e.g., income group, region, or risk category), the GATE for group \(g\) is:  

\begin{equation}
\delta_g \;=\; E[\,Y_i(1) - Y_i(0)\mid G_i = g\,]
\end{equation}

which represents the expected treatment effect for individuals in group \(g\). GATE is essentially a conditional ATE for a well-defined subset of the population.  

Parametric estimation of GATE is straightforward when group membership is observable. One can estimate separate treatment effects within each group or include treatment–group interaction dummies in a regression. For instance, with a binary group indicator \(G_i \in \{0,1\}\), one could extend the earlier model to:  

\begin{equation}
Y_i = \beta_0 + \beta_1 D_i + \beta_2 G_i + \gamma\, D_i \cdot G_i + f({x}_{i}){\prime}\boldsymbol{\beta} + \varepsilon_i
\end{equation}

Here, \(\beta_1\) is the treatment effect for the baseline group (\(G_i=0\)), and \(\gamma\) captures the additional effect for the group with \(G_i=1\). Thus, \(\delta_{G=0} = \beta_1\) and \(\delta_{G=1} = \beta_1 + \gamma\). More generally, with multiple groups, one can include a full set of interactions \(D \times I\{G=g\}\) for each group \(g\) to estimate each \(\delta_g\) (often with one group as a reference).  

If treatment is randomly assigned (or ignorably assigned after controlling for covariates), the difference in mean outcomes between treated and control units within each group provides an unbiased estimator of \(\delta_g\). Empirically, this is equivalent to stratifying the sample by group and computing the difference in outcomes—a method widely used in social sciences for moderation analysis.  

#### **Separate Regressions**  

A widely used and related approach to interactions is to conduct subgroup analyses by stratifying the sample. This means estimating the treatment effect separately within levels of a categorical variable or within slices of a continuous variable (e.g., by age group, by gender, by income quantile, by health status).  

Subgroup analysis involves running separate regressions for each subgroup:  

\begin{equation}
Y_i = \beta_0^g + \beta_1^g D_i + f(x_i){\prime}\boldsymbol{\beta} + \varepsilon_i, \quad \text{for each group } g
\end{equation}

For instance, researchers might report an estimated treatment effect (and confidence interval) for men and for women separately, then perform a statistical test for interaction to see if those effects differ significantly.  

These two approaches—interaction models and separate regressions—are intuitive and easy to communicate, but they are not identical and must be chosen carefully. In a single regression model with group-treatment interactions, the coefficients of all covariates that are not interacted with treatment remain constant across individuals, implying a homogeneity assumption in how these covariates affect outcomes across groups. Standard errors are also pooled accordingly.  

In contrast, separate regressions allow all coefficients, including those of control variables, to vary across groups. This distinction matters because it can lead to different conclusions. If covariates influence groups differently or if omitted variables affect groups in distinct ways, a single regression model may impose misleading restrictions, while separate regressions naturally accommodate group-specific effects.  

Thus, the choice between these methods should be made carefully, considering whether the homogeneity assumption of the single regression is appropriate for the given context. If the goal is to identify systematic differences in treatment effects across groups, separate regressions may be more suitable. However, if the focus is on a specific subgroup or on interactions with a particular covariate, a single regression with interactions may be more appropriate.

Here is a R simulation to illustrate the difference between a single regression with interaction terms and separate regressions for each group. The key idea is to showing a synthetic dataset where a treatment effect and covariates effect groups differently, then compare the estimated coefficients and standard errors across the two approaches.


``` r
# Load required packages
library(estimatr)  # For lm_robust()
library(dplyr)
# Set seed for reproducibility
set.seed(42)
# Generate simulated data
n <- 1000  # Total sample size
group <- sample(c(0,1), n, replace=TRUE)  # Binary group indicator
treatment <- sample(c(0,1), n, replace=TRUE)  # Treatment assignment
x <- rnorm(n)  # Covariate
error <- rnorm(n)  # Random noise

# Define true treatment effects for each group
beta1_g0 <- 2  # Treatment effect for Group 0 
beta1_g1 <- 4  # Treatment effect for Group 1
beta_x_g0 <- 1  # Effect of x for Group 0
beta_x_g1 <- 2  # Effect of x for Group 1

# Generate outcome variable with different effects by group
y <- ifelse(group == 0, 
            5 + beta1_g0 * treatment + beta_x_g0 * x + error, 
            8 + beta1_g1 * treatment + beta_x_g1 * x + error)

# Create data frame
data <- data.frame(y, treatment, group, x)

# Single regression with interaction terms using robust standard errors
single_reg <- lm_robust(y ~ treatment * group + x, data=data)
#summary(single_reg)

# Extract coefficients
b_treat <- coef(single_reg)["treatment"]  # Treatment effect for group 0
b_interact <- coef(single_reg)["treatment:group"]  # Interaction term
b_treat_g1 <- b_treat + b_interact  # Treatment effect for group 1

# Separate regressions for each group using robust standard errors
reg_group0 <- lm_robust(y ~ treatment + x, data = data %>% filter(group == 0))
reg_group1 <- lm_robust(y ~ treatment + x, data = data %>% filter(group == 1))
#summary(reg_group0)
#summary(reg_group1)

# Compare estimated coefficients
coef_comparison <- data.frame(
  Method = c("Single Regression (G=0)", "Single Regression (G=1)", 
             "Separate Regression (G=0)", "Separate Regression (G=1)"),
  Treatment_Effect = c(b_treat, b_treat_g1, coef(reg_group0)["treatment"], 
                       coef(reg_group1)["treatment"]),
  X_Effect = c(coef(single_reg)["x"], coef(single_reg)["x"], 
               coef(reg_group0)["x"], coef(reg_group1)["x"])
)
print(coef_comparison)
```

```
##                      Method Treatment_Effect  X_Effect
## 1   Single Regression (G=0)         1.984070 1.4705540
## 2   Single Regression (G=1)         4.092837 1.4705540
## 3 Separate Regression (G=0)         2.020198 0.9947716
## 4 Separate Regression (G=1)         4.003187 1.9645039
```

The treatment effect for the baseline group (G=0) in the single regression is represented solely by the treatment coefficient, while the effect for group 1 is computed as the sum of the treatment and the treatment-by-group interaction coefficients. As seen, the treatment effects for both groups in the single regression and the corresponding effects from separate regressions are similar but not identical. Additionally, while the coefficient for x remains constant in the single regression, it varies between groups in the separate regressions, reflecting distinct effects of x across subgroups. This example illustrates the importance of understanding the implications of each approach and choosing the most appropriate method based on the research question and context.

Unfortunately, it is well known that forking paths (data forking) can lead to data snooping issues, where researchers test multiple analytical choices—such as variable selection, model specification, or subgroup definitions—until they find a significant result, increasing the risk of false discoveries. The same issue applies when identifying subgroups – if many groups are examined, some may show spurious differences. To address this, in clinical trials, important subgroups are often defined *a priori* (e.g. by disease severity) based on mechanism and plausibility and analysts check whether the treatment’s efficacy is consistent across these groups.[^readext] In development economics, randomized controlled trials (RCTs) often adopt a similar approach by preregistering subgroup analyses based on theoretically motivated stratifications, such as household wealth or baseline education levels, to ensure that findings are not driven by post-hoc data mining.[^extpre] Preregistration helps mitigate concerns of multiple hypothesis testing and publication bias by committing researchers to planned analyses before observing outcomes. In both medical and development RCTs, transparent reporting of heterogeneous effects strengthens the credibility of policy-relevant conclusions. If many subgroups are explored without pre-specification, there is a risk of finding false positives (spurious heterogeneity), so corrections or validation in hold-out samples are advised.

[^readext]: Read "Developing a Protocol for Observational Comparative Effectiveness Research: A User's Guide, Chapter 3" by Varathan et al. (2013) for a detailed discussion on this topic.

[^extpre]:Read randomized control trials registration in economics from https://docs.socialscienceregistry.org/

Researchers, especially in health research, often distinguish *quantitative* interactions (treatment effects differ in magnitude across subgroups but have the same sign) from *qualitative* or *crossover* interactions (treatment helps one group but harms another). Detecting qualitative interactions is particularly important for policy – e.g. finding that a program benefits low-income individuals but has negative effects for high-income individuals would signal a need for targeted implementation.[^readext2]

[^readext2]: Read "Considerations when assessing heterogeneity of treatment effect in patient-centered outcomes research" by Lesko et al. (2018) for a detailed discussion on this topic.

Subgroup analyses on their own do not adjust for multiple confounders simultaneously (they typically split by one variable at a time), but when combined with regression adjustment, they can be powerful for communicating heterogeneity in an accessible way (e.g. a table of treatment effects by region, or a forest plot of effects in various patient subpopulations). Subgroup analysis is easy to interpret for policymakers (who can say “the program works for group A but not for group B”), but each subgroup estimate is effectively an average for that group and may miss continuous gradations in effect. 

### Parametric Generalized Linear Models (GLMs) 

Beyond linear regression, parametric outcome models can incorporate interactions to allow treatment effect heterogeneity. For example, in a logistic regression model (a GLM for binary outcomes with a logit link), a treatment-by-covariate interaction ($D \times X$) enables treatment effects to vary by subgroup. Similarly, in survival analysis (Cox proportional hazards models), an interaction of treatment with a covariate allows hazard ratios to change across groups. These parametric models are particularly useful when the outcome is non-linear (binary, count, survival time), ensuring an appropriate model while capturing heterogeneity.  

A key consideration is estimating the marginal treatment effect (MTE) rather than just the ATE. While ATE gives a population-wide estimate, MTE describes how the treatment effect changes with observable characteristics, making it useful for assessing effect heterogeneity.   

**Example: Marginal Treatment Effect in a Logistic Regression**  
Consider a binary outcome $Y$ (e.g., employment) and a treatment $D$ (e.g., receiving job training). We allow treatment effects to vary by years of education $X$. A logistic regression model: 

\begin{equation}
\log\left(\frac{P(Y=1)}{1-P(Y=1)}\right) = \beta_0 + \beta_1 D + \beta_2 X + \beta_3 (D \times X)
\end{equation}

The term $\beta_3$ captures how treatment effects vary by education level. The **marginal treatment effect** at a given $X$ is computed as the partial derivative:  

\begin{equation}
\frac{\partial P(Y=1 | D, X)}{\partial D}
\end{equation} 

This provides insight into how treatment effects change across different values of $X$. If $\beta_3 > 0$, treatment is more effective for higher-educated individuals. Keep in mind, in a linear model, the marginal effect is simply the estimated coefficient of the independent variable, since the relationship is additive and constant.

**Software Implementation**  

- **Stata**: Use `logit` or `logistic` with interaction terms, then apply `margins` for marginal effects.[^statsour] For penalized models, `lasso logit` can be used.  
  ```stata
  logit employment training education c.training#c.education
  margins, dydx(training) at(education=(10(2)20))
  ```
 
 [^statsour]:For explanation and various examples read https://www.stata.com/manuals/rmargins.pdf
 
- **R**: Use `glm()` for logistic models and `margins` package for marginal effects.[^rsour]  
  ```r
  model <- glm(Y ~ D * X, family = binomial, data = mydata)
  library(margins)
  marginal_effects <- margins(model, variables = "D")
  summary(marginal_effects)
  ```
  For penalized models, `glmnet` can select relevant interactions.

[^rsour]:For explanation and various examples read https://cran.r-project.org/web/packages/margins/vignettes/Introduction.html

- **Python**: Use `statsmodels` for logistic regression and `marginaleffects` for marginal effects.  
  ```python
  import statsmodels.api as sm
  from statsmodels.formula.api import logit
  model = logit("Y ~ D * X", data=df).fit()
  print(model.summary())
  ```
  For penalization, `sklearn.linear_model.LogisticRegressionCV` with `l1` penalty (Lasso) can be used.

Davis and Heller (2019) illustrate how standard researcher-specified interactions to search for subgroup heterogeneity methods can miss nuanced heterogeneity. They evaluated a summer jobs program for youth, implementing both the standard interaction model and causal forest, a machine learning approach which we will cover below. They demonstrate that the causal forest predictions successfully identify heterogeneity in employment treatment effects that typical interaction effects would miss. Highly recommend not only reading the paper but also reviewing their very good supplemental document, which provides a clear comparison of standard heterogeneity methods and this new approach, making it an insightful and accessible resource.

### Random Coefficients Models 
A more flexible parametric approach involves **random coefficient models**, where treatment effects vary across individuals but follow an assumed distribution:

\begin{equation}
Y_i = \beta_0 + (\beta_1 + u_i) D_i + \beta_2 X_i + \varepsilon_i, \quad u_i \sim \mathcal{N}(0, \sigma^2)
\end{equation}

Here, \( u_i \) represents individual-specific deviations from the average treatment effect, allowing for unobserved heterogeneity. These models are useful when heterogeneity is thought to arise from latent factors not captured in the observed covariates.

**Advantages and Limitations of Parametric Approaches** 

 The theoretical underpinning in these models is that if the specified interactions capture the true effect, one can consistently estimate subgroup-specific effects by ordinary least squares or maximum likelihood. Parametric methods provide interpretable estimates and require fewer data than fully nonparametric or machine learning-based approaches. In practice, parametric models require the researcher to pre-specify which interactions (or sub-groups) to examine. 
 
If the parametric form is correct, interaction models provide unbiased and efficient estimates of subgroup effects and are easy to interpret. Classical hypothesis tests on interaction coefficients inform whether heterogeneity along that covariate is statistically significant. However, a key limitation is potential misspecification: if the true heterogeneity is more complex (non-linear or involving higher-order interactions), a simple linear interaction can misestimate the true pattern of (x). Misspecification can lead to biased estimates of treatment heterogeneity. Additionally, these models often require careful consideration of which covariates to include and how to model interactions to avoid omitted variable bias or overfitting. Regularization techniques (like Lasso) can be used to select a sparse set of important interactions, as introduced in earlier chapters. This helps mitigate the curse of dimensionality in purely parametric models by shrinking unimportant interaction effects towards zero.

Despite these challenges, parametric HTE models remain popular for their simplicity. They are often the first approach economists or health researchers try when looking for subgroup differences. They are simple and interpretable. Their limitations (risk of model misspecification and limited ability to capture complex interactions) inspire the use of the more advanced methods discussed below, but they set the stage for understanding what HTE means substantively (e.g. identifying which observable factors might make a difference in treatment response). Let's explore semi-parametric techniques in the following section. 

## Double Machine Learning for Conditional Average Treatment Effects (CATE)  

Earlier, we discussed parametric methods for estimating heterogeneous treatment effects, including interaction models and subgroup analysis. While these approaches provide interpretability, they rely on strong functional form assumptions that may overlook complex treatment heterogeneity. We then introduced parametric GATE, which examines treatment variation at the subgroup level. However, predefined group structures may not fully exploit the richness of available data, and estimating interactions in high-dimensional settings can be challenging.  

Purely parametric models risk misspecification when heterogeneity is complex. Semiparametric methods offer a middle ground by using machine learning to flexibly estimate nuisance functions—such as regression or propensity scores—while targeting a low-dimensional, structured effect. DML for CATE provides a flexible approach that integrates these techniques while preserving valid statistical inference. Built on Neyman orthogonality and cross-fitting, DML mitigates overfitting and bias, ensuring robust estimation of heterogeneous treatment effects even in high-dimensional settings. Traditional parametric methods, such as regression models with interaction terms, often fall short when dealing with high-dimensional confounders, nonlinearities, and complex interactions that influence both treatment assignment and outcomes. Model misspecification can introduce bias, limiting the accuracy of treatment heterogeneity estimates. DML provides a more robust alternative for studying treatment effect heterogeneity. We also note that related approaches like targeted maximum likelihood estimation (TMLE) and partial linear models with ML fall in this category, but DML has become especially influential in economics.

**The DML Framework for CATE Estimation** 

DML extends the standard partially linear model for treatment effects (\(Y_i =  \delta D_i + f(X_i)  + \varepsilon_i\)) we covered in chapter 19 by incorporating heterogeneity. The key theoretical idea behind DML is to set up an estimation problem with orthogonal moment conditions. In a simplified setting, suppose the outcome can be decomposed as

\begin{equation}
Y_i = g(X_i) + \delta(X_i).D_i + \varepsilon_i
\end{equation}

where \( Y_i \) is the outcome, \( D_i \) is the binary treatment indicator, \( X_i \) represents observed covariates, \( \varepsilon_i \) is the error term. Moreover,  $g(X_i) = E[Y_i \mid X_i]$ captures how baseline outcomes vary with covariates and $\delta(X_i) = E[Y(1)-Y(0)\mid X_i]$ is the true heterogeneous treatment effect function we aim to learn. Both $g(\cdot)$ and $\delta(\cdot)$ are a priori unknown and potentially complex, need to be estimated. Using machine learning to estimate these components can introduce bias, but DML corrects for this using Neyman orthogonality and cross-fitting. If we also define $e(X_i) = P(D_i=1 \mid X_i)$ as the propensity score (for treatment), the following moment condition holds at the true functions:

 \begin{equation}
    E\Big[\big(Y_i - g(X_i) - \delta(X_i)(D_i - e(X_i))\big)(D_i - e(X_i))\Big] = 0
    \end{equation}
    
Intuitively, this Neyman orthogonality condition says that if we remove the systematic outcome variation due to $X_i$ (through $g$) and center the treatment assignment (through $e$), then any remaining correlation between those residuals indicates a mis-specified $\delta(\cdot)$. In other words, $\delta(X)$ can be identified as the function that makes the residual outcome uncorrelated with the residual treatment, conditional on $X$. This orthogonality property is powerful: it implies that small estimation errors in $g(\cdot)$ or $e(\cdot)$ (the “nuisance” functions) will not heavily bias the estimation of $\delta(\cdot)$, at least to first order (Neyman orthogonality).

In practice, DML for CATE proceeds through the following steps:

**Step 1: Estimating Nuisance Functions (First Stage)**  
DML first estimates two nuisance functions using flexible econometrics and ML methods (e.g., logistic regressions, random forests, gradient boosting (XGBoost), and neural networks):  
- **Outcome model**: \( \hat{g}(X) = \mathbb{E}[Y \mid X] \), estimated via a predictive classifier or regression model.  
- **Propensity score**: \( \hat{e}(X) = P(D=1 \mid X) \), also estimated using a predictive regression model.  

Since ML methods tend to **overfit**, directly plugging these estimates into treatment effect models leads to biased estimates. To address this, DML applies cross-fitting.

**Step 2: Cross-Fitting to Reduce Bias**  
Instead of using the same data for estimating nuisance functions and treatment effects, cross-fitting splits the data into \( K \) folds:  
1. Train \( \hat{g}(X) \) and \( \hat{e}(X) \) on one subset.  
2. Estimate treatment effects on a separate subset to prevent overfitting.  
3. Rotate the subsets, repeating the process across all folds.  
Cross-fitting prevents overfitting by separating nuisance estimation from CATE estimation.  

**Step 3: Orthogonalized Score Function (Second Stage Estimation)**  
Using the cross-fitted nuisance estimates, we construct residualized versions of the outcome and treatment:

\[
\tilde{Y} = Y - \hat{g}(X)
\]

\[
\tilde{D} = D - \hat{e}(X)
\]

Once the residuals \((\tilde{Y}, \tilde{D})\) are computed, we estimate the heterogeneous treatment effect function \(\tau(X)\) by solving:

  \[
    \tilde{Y} = \tau(X) \tilde{D} + \varepsilon
    \]
    
This regression is performed usually using OLS to capture heterogeneity in \(\tau(X)\).

In this way, we obtain an orthogonalized estimator of \( \tau(X) \):

\begin{equation}
\tau(X) = \arg\min_{\tau} \sum_{i} \left( \tilde{Y}_i - \delta(X_i) \tilde{D}_i \right)^2
\end{equation}

Since \( \tilde{D} \) is **orthogonal** to errors in \( \hat{g}(X) \) and \( \hat{e}(X) \), this approach removes bias introduced by machine learning errors. 

The DML approach has attractive theoretical guarantees. Under regularity conditions, the estimated $\hat\delta(x)$ is asymptotically unbiased and can achieve approximate normality, enabling confidence intervals for functionals of $\delta(x)$ or for $\delta(x)$ at a point with Neyman orthogonality. The use of cross-fitting is crucial here: it breaks the dependence between the estimated nuisances and the individual data points used to estimate $\delta$, thereby satisfying the orthogonality condition and enabling $\sqrt{n}$-consistent estimation even when using very flexible learners for $g$ and $e$ (Chernozhukov et al., 2018).

Despite its complexity, DML has become feasible thanks to modern computing and has been implemented in various software (e.g., the DoubleML package in Python and R). However, there are computational challenges: one must train multiple machine learning models and iterate across folds, which can be intensive for large datasets. Tuning the learners for $g$ and $e$ is also critical – if either model is very poor, variance can inflate (though bias remains controlled by orthogonality). Moreover, when the covariate space is high-dimensional, one might still need to impose some structure on $\delta(x)$ (such as modeling $\delta(x)$ with a simpler ML method or basis expansion) to avoid the curse of dimensionality.

Meta-Learners like the R-learner and DR-learner provide an alternative approach to estimating heterogeneous treatment effects, which we will discuss in the next chapter. Unlike DML, which emphasizes orthogonalization for valid statistical inference, meta-learners focus on flexible, nonparametric estimation of treatment heterogeneity but is not primarily designed for inference. These methods trade off interpretability for adaptability in capturing complex treatment variation. 

Software implementations for these methods are readily available: the DoubleML package in R/Python provides a high-level interface for DML, and Microsoft’s EconML library in Python offers R-learner and other metalearners for CATE with built-in cross-fitting. Researchers comfortable with Stata can still incorporate these techniques by exporting prediction results from ML models (e.g. use Python/R to get $\hat e(X)$, then use Stata for the orthogonalized regression).

Semiparametric methods still impose some structure (e.g. a specified moment condition or basis). In contrast, fully nonparametric approaches aim to let the data speak as freely as possible about heterogeneity in \(\delta\) , with minimal functional assumptions. We will explore these methods in the next section.

## Fully Nonparametric Methods for HTE Estimation  

Fully nonparametric approaches impose minimal structural assumptions on how treatment effects vary with covariates. Instead of specifying a functional form or a small set of interactions, these methods allow the data to reveal complex heterogeneity. Traditional techniques include nonparametric smoothing, matching, and weighting, alongside modern causal machine learning methods like recursive partitioning (causal trees) and ensemble methods (causal forests). These methods can approximate any heterogeneity pattern given sufficient data, though they may suffer from interpretability challenges and higher variance. Below, we discuss key methods and recent applications.

### Matching and Weighting Methods for HTE Estimation 

Matching and weighting are classic nonparametric causal inference techniques,we covered in Chapter 20, adapted here to estimate heterogeneous effects. The basic idea is to compare treated and control units who are similar in covariates to get an estimate of the treatment effect for those covariate profiles. Instead of fitting a global model, matching/weighting construct local approximations of the counterfactual. For heterogeneous effects, one can do matching within subgroups or even at the individual level.

**Matching-Based HTE Estimation** method pairs treated and control units with similar observed characteristics to estimate treatment effects as we covered in Chapter 20 . It relies on the assumption that treatment assignment is as good as random after conditioning on observables. HTE estimation via matching involves estimating treatment effects separately for subgroups or individuals. Common techniques include:

- Nearest Neighbor Matching: Matches each treated unit to the closest control unit in covariate space.

- Propensity Score Matching: Matches units with similar estimated treatment probabilities.

- Genetic Matching: Uses optimization to find optimal covariate balance.

To estimate GATE, matching can be performed separately for each subgroup, allowing for treatment effects to be estimated nonparametrically at the group level.  

\begin{equation}
\tau_g = \frac{1}{N_g} \sum_{i \in g} \left( Y_i(1) - Y_i(0) \right),
\end{equation}

where \( N_g \) is the number of matched units in group \( g \). 

**Propensity Score Weighting (IPW):** Propensity score methods typically aim at ATE, but we can modify them to get CATE. For instance, to estimate the treatment effect for a specific subgroup defined by $X=x$, one could use inverse probability weights that emphasize observations near $x$. In practice, one might stratify the sample by the propensity score (which is a function of $X$) – this is essentially subclassification. Within each propensity score stratum (which groups units with similar $X$), we can estimate an effect. If those strata are narrow, each yields an approximately homogeneous effect estimate; by looking at how the estimate changes across strata, we observe heterogeneity. 

Another approach is **caliper or kernel matching on the propensity score**: for a given treated unit $i$, find control units with close propensity scores (or covariates) and compute the difference in outcomes; that gives an individual-level estimate $\hat{\tau}_i$. One can then aggregate those $\hat{\tau}_i$ for those with similar $X$ to see patterns. Matching on the *covariates* themselves (rather than the propensity) is also used: e.g. exact matching on some key discrete attributes (like matching each treated woman with a similar control woman) ensures heterogeneity by that attribute is accounted for without model assumptions.

In summary, matching and weighting methods can estimate HTE by isolating comparable groups and computing treatment effects within them. They are nonparametric in spirit – no functional form, just pairwise comparisons – but they can be combined with parametric elements (like a propensity score model, which is usually logistic = parametric, although one can use boosted trees to estimate propensity = semi-parametric). These approaches are intuitive and complement machine learning: indeed, some modern ML algorithms for HTE, like *causal forests*, can be seen as an automated way of doing many localized matchings or weightings (with the forest learning the appropriate neighborhoods for averaging).

### Kernel-Based Methods for HTE Estimation

One direct approach to estimate a CATE function is through kernel smoothing or localized regression. Kernel method is a smoothing approach where weights are assigned based on the similarity of observations, typically using kernel functions. In the context of HTE, kernel regression can be used to directly estimate treatment effects continuously across covariates, rather than forming discrete matches. Essentially, one can run two nonparametric regressions: $\hat{\mu}_1(x) = \frac{\sum_{i: D_i=1} K_h(X_i-x) Y_i}{\sum_{i: D_i=1} K_h(X_i-x)}$ and $\hat{\mu}_0(x) = \frac{\sum_{i: D_i=0} K_h(X_i-x) Y_i}{\sum_{i: D_i=0} K_h(X_i-x)}$, where $K_h$ is a kernel weighting function around $x$. The CATE is then $\hat{\tau}(x) = \hat{\mu}_1(x) - \hat{\mu}_0(x)$. 

This is a fully nonparametric estimator of the treatment effect function, using local averaging. For example, if $X$ is one-dimensional (say age), this reduces to the idea of estimating the treatment effect at each age by comparing outcomes of treated and control individuals near that age (perhaps using a Gaussian kernel weight). If $X$ is high-dimensional, kernels suffer from the curse of dimensionality, so in practice one might use dimensionality reduction or only a subset of covariates for the kernel. Nonetheless, this approach has been used in some policy evaluations; for instance, Ichimura and Heckman’s work in the 1990s on active labor programs used kernel weighting to estimate distributional impacts. In the past 5–10 years, less manual kernel smoothing is done (since tree-based methods have become popular), but some recent biostatistics papers have revisited kernel methods to estimate CATE because they are straightforward to analyze theoretically. 

An extension is local linear matching: fit a local linear model within a neighborhood of $x$, which can reduce bias at boundaries. These methods are nonparametric in that they make no global assumption about $\tau(x)$; however, they require large data to get stable estimates, and confidence intervals often rely on asymptotic approximations. Software-wise, one can implement kernel CATE in R with packages like `np` (nonparametric kernel methods) or even by writing a short routine. 

Kernel methods can also estimate GATEs by smoothing treatment effects within a neighborhood of $X=x_0$”, producing local GATE estimates. Instead of hard-threshold stratification, kernel smoothing provides a nuanced view of treatment effect variation. In some applications, such as epidemiology, kernel density estimation helps define groups based on continuous risk factors (e.g., blood pressure), avoiding arbitrary cutoff. However, one can also form actual discrete groups by clustering $X$ values via kernel methods. For example, use kernel density estimation on $X$ to find modes (clusters of covariate space), then treat those clusters as groups and estimate GATEs for each.

From a methodological standpoint, kernel-based estimation of heterogeneity is statistically well-understood (it falls under nonparametric regression), but in practice it’s less used in recent empirical work compared to trees/forests, likely because trees are easier to interpret. That said, some recent papers do use BART (Bayesian Additive Regression Trees) which is akin to a kernel method in that it’s a sum-of-trees model that can capture smooth nonlinear heterogeneity. BART has been used for personalized medicine studies to estimate patient-specific treatment effects and then those are averaged for groups. Yet, we will cover BART in this book as it is mainly a Bayesian method. 

In sum, kernel-based GATE is a more technical approach and not as frequently highlighted in economics literature by name. It resides in the toolbox of statisticians. It’s important to remember it as a link between fully parametric subgroup analysis and black-box ML: one can always do something like “loess” (locally weighted regression) to map out treatment effect heterogeneity, then communicate it in terms of groups (“the treatment effect is about 2 units at X=10, and about 5 units at X=20”). For completeness, we include it in this section, but the star methods for fully nonparametric HTE in economics and related fields have been the tree-based algorithms, to which we turn next.





