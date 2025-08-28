# Selection on Observables

While randomized controlled trials (RCTs) remain the gold standard for establishing causal relationships, they are often impractical or unfeasible in real-world settings due to several challenges we discussed in the previous chapter. Moreover, non-compliance by administrators, where experimental protocols are not strictly followed, can undermine randomization. Similarly, non-compliance among treated individuals, such as those failing to adhere to the intervention, or among controls, who may seek out the treatment despite being in the control group, can distort the validity of causal estimates. In such cases, researchers must turn to alternatives that leverage observational data to approximate the conditions of a controlled experiment.

Randomization ensures that treatment assignment is independent of potential outcomes, eliminating selection bias and enabling straightforward causal inference. However, in observational data, treated and untreated groups often differ systematically in observed characteristics, introducing selection bias. To address this, researchers rely on the assumption that treatment assignment is independent of potential outcomes after adjusting for certain covariates (\(X\)), effectively making the assignment “as good as random” within groups sharing the same values of \(X\). This approach allows researchers to infer causal effects from observational data by mimicking the conditions of randomization, provided that all relevant confounders are accounted for. Under the selection on observables framework, researchers use methods such as matching or regression adjustment for observable differences between treated and untreated groups. This framework relies on the assumption of conditional independence and overlap to estimate causal effects. 

Starting with this chapter and continuing over the next three, we examine the selection on observables framework, which assumes that treatment assignment is independent of potential outcomes after conditioning on observed covariates. We begin by outlining the key identification assumptions—conditional independence and common support—required for valid causal inference. The next chapter then explores various regression estimation methods, including regression adjustment and machine learning-based approaches such as double/debiased LASSO in detail. Through theoretical discussions, empirical examples, and simulations, this chapter provides a structured guide for estimating treatment effects in observational data settings using regression. In the next chapters, we will also cover methods such stratification, matching, propensity score, inverse probability weighting, and doubly robust estimators. 

We want to emphasize that selection on observables methods rely on the assumption that treated and untreated units differ only in ways we can observe, making causal effect estimates from these methods open to debate and potentially less credible. While these approaches are data-driven, widely used in econometrics and machine learning, and valuable for practical applications, it is crucial to recognize that in economics, social sciences, and health research, where human behavior is central, countless unobservable factors influence actions and decisions, limiting our ability to control for all relevant variables. When selection on observables is not sufficient because unobserved factors also influence treatment assignment and outcomes, researchers may rely on methods that address selection on unobservables. Following these three chapters, we focus on models designed to handle unobservable factors.

## Assumptions and Definitions Behind Selection on Observables

Our main goal is to estimate the Average Treatment Effect (ATE), but we can also estimate measures like ATT and ATU as causal effects. Estimating causal effects under Selection on Observables relies on two key assumptions. These assumptions are crucial for ensuring that treatment assignment is "as good as random", allowing researchers to isolate the causal effect of treatment from confounding factors:

### Conditional Independence Assumption (Unconfoundedness)
   “Selection on observables” assumption posits that treatment assignment is independent of the potential outcomes, \( (Y_i(1), Y_i(0)) \), once we condition on observed covariates, \( X_i \). Mathematically, this is expressed as:
   
\begin{equation}
   (Y_i(1), Y_i(0)) \perp D_i \mid X_i
\end{equation}
   
   where \( D_i = 1 \) if the individual receives treatment and \( D_i = 0 \) otherwise. This implies that any systematic differences in outcomes between treated and untreated/control groups can be fully explained by differences in \( X_i \), eliminating the influence of confounding variables.

   For example, in a scholarship study where \(X_i\) covers parental income, university type, and SAT scores, individuals with identical \(X_i\) have the same chance of receiving a scholarship, independent of their unobserved earning potential. Similarly, in a study of job training programs, if \(X_i\) includes age, income, and education level, SOO assumes that individuals with the same \(X_i\) are equally likely to receive treatment regardless of their potential outcomes. In health research, when \(X_i\) consists of age, pre-existing conditions, and socioeconomic status, SOO assumes that people with the same \(X_i\) are equally likely to participate in a preventive care program, regardless of factors not captured in \(X_i\).

  In essence, conditional independence assumption (CIA) tells us that:

- Potential outcomes are independent of \( D_i \), conditional on covariates.

- This is often referred to as conditional unconfoundedness or conditional independence.

- In simpler terms, once we control for \( X_i \) (observed covariates), treatment assignment is as good as random.

- Practically, controlling for \( X_i \) eliminates selection bias, making causal interpretation plausible under SOO methods.

### Common Support Assumption (Overlap Condition)
   The overlap assumption ensures that every individual has a non-zero probability of being either treated or untreated, given their covariates. Formally, this is expressed as

\begin{equation}
0 < P(D_i = 1 \mid X_i = x) < 1, \quad \forall x
\end{equation}

This condition guarantees that both treated and untreated units are available for each level of \(X_i\), allowing meaningful comparisons. Without overlap, it becomes challenging to compare units with similar covariates, and standard selection-on-observables methods (which we will cover in this chapter) may yield invalid causal estimates. However, alternative approaches—such as regression discontinuity and instrumental variables (which we will cover in the next chapter)—can sometimes identify causal effects even when the overlap assumption does not hold.

## Defining the (Conditional) Average Treatment Effect

\begin{itshape}
Why did CATE leave ATE at the altar? 

Because ATE kept averaging out their differences, and CATE was tired of all the behind-the-scenes covariate adjustments!
\end{itshape}

In the previous section on randomization, we demonstrated that under the identification assumption of random assignment, selection bias is eliminated, providing an unbiased estimate of the Average Treatment Effect (ATE). Specifically:

\begin{equation}
\delta^{ATE} = E[Y_i(1) - Y_i(0)] = E[Y_i(1)] - E[Y_i(0)] = E[Y_i \mid D = 1] - E[Y_i \mid D = 0]
\end{equation}

This simple difference in means of observable outcomes between the treated and untreated groups in observed data serves as an estimate of the ATE. 

When treatment assignment is not random and depends on observable characteristics, the Conditional Independence Assumption (CIA) and common support condition allow us to estimate the Conditional Average Treatment Effect (CATE) for any given covariate \(X_i\). By averaging these CATEs over the distribution of \(X_i\) in the population, we can recover the average causal/treatment effect, \(\text{ATE} = \mathbb{E}[\text{CATE}(X_i)]\). This framework ensures causal estimation is possible even in selection in observable settings, provided the key assumptions hold.

Under the Selection on Observables (SOO) framework, the CIA assumption implies:

\begin{equation}
E[Y_i(1) - Y_i(0) \mid X_i] = E[Y_i(1) \mid X_i, D_i = 1] - E[Y_i(0) \mid X_i, D_i = 0]
\end{equation}

This means that within groups with the same values of \(X_i\), treatment assignment is "as good as random." For a specific value of \(X_i = x\), we can define the Conditional Average Treatment Effect (CATE) as:

\begin{equation}
\text{CATE}(x) = E[Y_i(1) - Y_i(0) \mid X_i = x] = E[Y_i \mid X_i = x, D_i = 1] - E[Y_i \mid X_i = x, D_i = 0]
\end{equation}

Under the common support condition, where all values of \(X_i\) are observed in both treated and untreated groups, we can aggregate the CATE across the population to estimate the ATE:

\begin{equation}
\delta^{ATE} = \sum_x \text{CATE}(x) P(X_i = x)
\end{equation}

Similarly, we can calculate the Average Treatment Effect on the Treated (ATT) and the Average Treatment Effect on the Controls (ATC) using the distributions of \(X_i\) in the treated and control groups:

\begin{equation}
\delta^{ATT} = \sum_x \text{CATE}(x) P(X_i = x \mid D_i = 1), \quad
\delta^{ATC} = \sum_x \text{CATE}(x) P(X_i = x \mid D_i = 0)
\end{equation}

These weighted averages allow for robust causal estimation by leveraging the observed covariates and addressing biases in observational data.[^contcov]

[^contcov]: For continuous covariates, the ATE can be expressed as an integral \(\delta^{ATE} = \int \text{CATE}(x) f_{X_i}(x) \, dx,\) where \(f_{X_i}(x)\) is the density of \(X_i\). Similarly, \(\delta^{ATT}\) and \(\delta^{ATC}\) use \(f_{X_i \mid D_i=1}(x)\) and \(f_{X_i \mid D_i=0}(x)\), respectively.

These identification assumptions and result are common to various estimation methods, which we will cover below, for causal effects. Regression methods such as OLS and double-lasso adjusts for confounding by modeling the relationship between \(Y_i\), \(D_i\), and \(X_i\), while subclassification groups units into strata based on covariates \(X_i\) and estimates treatment effects within each stratum. Matching pairs treated and untreated units with similar \(X_i\) values. Other methods, such as inverse probability weighting (IPW), use propensity scores to reweight observations, ensuring comparability between treated and control groups. Doubly robust estimators combine regression and weighting for added robustness, while machine learning techniques, like causal forests, estimate CATEs flexibly and handle complex covariate structures. Spline or kernel methods offer smooth adjustments for \(X_i\) without assuming a specific functional form. These approaches differ in how they condition on \(X_i\) and weight to estimate CATEs, providing versatility for a range of data and study designs.


**Example:** To understand the concepts of ATE and CATE, let’s consider a study estimating the effect of receiving a college scholarship on annual earnings. Suppose the exact same amount of scholarships are allocated randomly among students,i.e. completely randomized experiment, ensuring that selection bias is removed. In this randomized setting, we observe that, on average, 25% of students receive the scholarship. The overall average annual earnings for students who received the scholarship (the treated group) are \$50,000, while for those who did not (the control group), the average earnings are \$40,000. The Average Treatment Effect (ATE) is calculated as:

\[
\text{ATE} = E[Y(1) - Y(0)] = E[Y \mid D = 1] - E[Y \mid D = 0] = 50,000 - 40,000 = 10,000
\]

This indicates that, on average, receiving a scholarship increases annual earnings by $10,000. Keep in mind, individual impacts can vary widely—some students may see little to no benefit, while others might experience tens of thousands of dollars in increased earnings. The ATE represents the average effect across all individuals, summarizing these various outcomes into a single measure.

Since randomization is not available, we rely on observational data, recognizing that multiple factors may influence earnings. However, for simplicity in this example, we assume university type (\(X_i\)) captures the key observable differences affecting both treatment assignment and outcomes. For example, private, public, and non-profit colleges might serve distinct populations, with varying effects of scholarships on earnings. Let’s calculate the CATEs for public and non-profit colleges, given the following data:

For public colleges, the average annual earnings of scholarship recipients are \$48,000, while for non-recipients, they are \$40,000. This gives a CATE for public colleges of:

\[
\text{CATE}_{\text{Public}} = E[Y(1) - Y(0) \mid X = \text{Public}] = 48,000 - 40,000 = 8,000
\]

For non-profit colleges, scholarship recipients earn an average of \$42,000 annually, compared to \$36,000 for non-recipients. The CATE for non-profit colleges is:

\[
\text{CATE}_{\text{Non-Profit}} = E[Y(1) - Y(0) \mid X = \text{Non-Profit}] = 42,000 - 36,000 = 6,000
\]

For private universities, scholarship recipients earn an average of \$65,000 annually, compared to \$50,000 for non-recipients. This gives a CATE for private universities of:

\[
\text{CATE}_{\text{Private}} = E[Y(1) - Y(0) \mid X = \text{Private}] = 65,000 - 50,000 = 15,000
\]

Assume 50% of students attend public colleges, 30% attend private universities, and 20% attend non-profit colleges. The ATE is then calculated as a weighted average of the CATEs:

\[
\text{ATE} = 0.5 \cdot 8,000 + 0.3 \cdot 12,000 + 0.2 \cdot 6,000 = 4,000 + 4,000 + 1,200 = 9,700
\]

This example illustrates how the ATE aggregates heterogeneous treatment effects (Conditional ATEs) across subgroups, weighted by their population shares. While the overall ATE suggests a $10,000 increase in earnings due to the scholarship, the CATEs highlight differences in impacts (average causal/treatment effect) by university type.  Students at private universities experience the largest gains, likely due to stronger networks or better resources, whereas public college students benefit moderately, and non-profit college students see smaller gains. These differences highlight the importance of considering heterogeneity in treatment effects to better inform policy decisions and target interventions more effectively.

In the following sections, we will explore various approaches under the selection on observables framework: regression methods such as OLS and double lasso, stratification and matching, propensity score, weighting approaches,and doubly robust method to estimate causal effects. These methods provide flexible tools for addressing the challenges of causal inference in observational data.

## Regression-Based Estimation Methods

In an observational setting, we adapt the same regression adjustment ideas from the randomized framework discussed in the previous chapter, but rely on the crucial assumption that treatment assignment is “as good as random” once we condition on all relevant, observed covariates. This section explains how regression adjustment helps us estimate causal effects when randomization is absent, clarifies how we move from conditional effects to an overall average treatment effect (ATE), and discusses issues of inference and robust standard errors. Conceptually, by using covariates that influence both treatment assignment and outcomes (confounding and pretreatment variables), regression adjustment (or any predictive modeling) effectively simulates random assignment within each covariate stratum. Once we account for these confounders, the treated and untreated groups become comparable in expectation, so the observed outcome differences reflect the average causal effect, and removes the selection bias. We illustrate these points using the scholarship example, yet emphasize that the approach applies generally.

In a randomized trial, treatment assignment is physically controlled, ensuring that—on average—treated and untreated groups only differ because of the intervention. In observational research, however, we cannot manipulate treatment directly. Instead, we assume that once we condition on observed characteristics \( X \), no unobserved confounders remain. This assumption, often termed "selection on observables," means that within each \( X = x \), the treatment is "as good as" random. Under this assumption, we can treat each subgroup \( X = x \) as if an internal experiment were carried out there, letting us compare treated and untreated outcomes within each subgroup.

When \( X \) is discrete (for instance, public vs. private colleges), it is straightforward to define subgroup-specific differences and then average them up. In the scholarship example, we separately compute the difference in earnings between treated and untreated students for public, private, and non-profit colleges, then weight each difference by the share of students in that subgroup to form an overall ATE. If \( X \) is continuous or contains multiple variables, the principle remains the same—though we replace that discrete summation with an integral over the distribution of \( X \). In practice, regression-based methods carry out this averaging automatically. Lets discuss two separate regression methods to estimate the ATE:

### Separate Regressions for Treated and Control Groups
A first approach, closely mirroring the randomized setting, is to fit two separate regressions—one for the treated group \((D_i = 1)\) and one for the untreated group \((D_i = 0)\). Concretely, we write:

\begin{equation}
Y_i = \gamma_1 + \beta_1^\top X_i + \varepsilon_i, 
\quad 
D_i = 1
\end{equation}

\begin{equation}
Y_i = \gamma_0 + \beta_0^\top X_i + \varepsilon_i, 
\quad 
D_i = 0
\end{equation}

Once these regressions are fitted, we obtain the fitted potential outcomes 
\(\hat{Y}(1 \mid X_i) = \hat{\gamma}_1 + \hat{\beta}_1^\top X_i\) and 
\(\hat{Y}(0 \mid X_i) = \hat{\gamma}_0 + \hat{\beta}_0^\top X_i\). For each individual \(i\) with covariates \(X_i\), the conditional average treatment effect (CATE) is 
\(\hat{Y}(1 \mid X_i) - \hat{Y}(0 \mid X_i)\). 

The overall average treatment effect (ATE) follows by averaging this individual-level difference across the entire sample:

\begin{equation}
\widehat{\text{ATE}} 
= 
\frac{1}{n} \sum_{i=1}^n 
\bigl[ 
\hat{Y}(1 \mid X_i) 
- 
\hat{Y}(0 \mid X_i) 
\bigr]
\end{equation}

When \(X\) is centered in each treatment group, the ATE can be read directly off the difference in intercepts \(\hat{\gamma}_1 - \hat{\gamma}_0\), just like in a randomized controlled trial (RCT) we discussed in the previous chapter. If \(X\) is not centered, the average difference over the sample still yields the correct ATE. Valid inference—through standard errors and confidence intervals—requires a robust variance estimate (e.g., Eicker-Huber-White) or bootstrapping - which is more common- to accommodate heteroskedasticity.

In separate regressions for regression adjustment in observational studies, we generate similar or nearly identical predicted adjusted outcomes for treated and control groups, effectively removing selection and omitted variable bias. Separate regressions naturally allow the form of \(Y\) on \(X\) to differ between treatment and control groups. This method is often used when you want maximum flexibility in modeling outcomes under each treatment status, and extends seamlessly to machine learning methods under selection on observables. If you believe all relevant confounders are observed in \(X\), you can adopt any predictive modeling strategy—whether a simple linear model, random forests, neural networks, or other algorithms—and fit it separately to the treated and untreated samples. Having done so, you predict an individual’s outcome assuming treatment,\(\hat{Y}(1 \mid X_i)\), using the “treated” model and individual’s outcome assuming treatment,\(\hat{Y}(0 \mid X_i)\), using the untreated model. The difference in those two predicted outcomes is that individual’s estimated treatment effect, \(\hat{\delta}_i = \hat{Y}(1 \mid X_i) - \hat{Y}(0 \mid X_i)\). Averaging these differences across \(i\) yields \(\widehat{\text{ATE}}\). In the machine-learning literature, this two-model approach is often called a T-learner, reflecting the idea that you train two separate models (one for each treatment state), which will be discussed in Meta-Learners chapter in detail.

### Single Regression with Covariate Adjustment

A common and straightforward way to estimate the average treatment effect (ATE) is through Ordinary Least Squares (OLS) regression. This method provides a direct and interpretable approach to causal effect estimation under the selection-on-observables assumption. It is important to emphasize again that this approach relies on the assumption that all relevant confounders are observed and included in \( X_i \). If unobserved confounders influence both treatment and outcomes, bias may still persist—necessitating alternative methods, which we will cover in the next chapter. In this approach, we specify the outcome model as: 

\begin{equation}
Y_i = \beta_D \cdot D_i + f(X_i) + \varepsilon_i
\end{equation}  

where \( Y \) is the outcome, \( D \) is the treatment indicator, \( \beta_D \) represents the treatment effect, and \( X \) includes all covariates whose influence on \( Y \) is captured by \( f(X)\). The OLS estimator for \( \beta_D \) provides an adjusted estimate of the ATE, denoted as \( \hat{\delta}^{ATE} = \hat{\beta}_D \). The derivation is identical to the case where treatment is randomized, except that here, adjustment for \( X \) accounts for selection-on-observables. Since standard OLS may overfit when \( X \) is high-dimensional, we later introduce other methods, such as debiased/double (ML) lasso. First, however, we discuss the single linear regression adjustment model, which serves as the baseline approach.  

\begin{equation}
Y_i = \alpha + \delta\, D_i + X_i^\top \beta + \varepsilon_i
\end{equation}  

Here, \( \delta \) is the difference between treated and untreated after partialing out the effect of \( X_i \). If no interaction terms are added, this model imposes a constant treatment effect—i.e., it treats \( \delta^{CATE}(x) = \delta \) for all \( x \). In that simplified world, the ATE is exactly \( \delta \). Mechanically, once you fit the regression, the difference in fitted outcomes between \( D=1 \) and \( D=0 \) is always \( \hat{\delta} \), irrespective of \( X_i \). Averaging over \( i \) in the sample therefore yields the same \( \hat{\delta} \). This approach can be viewed as "automatically" integrating (or summing) the effect across all \( X_i \) values in the data. In other words, the fitted model simultaneously estimates the conditional mean of \( Y \) for treated and untreated units across the full range of \( X_i \), then (under the selection-on-observables assumption) “averages” those conditional means over the observed distribution of \( X \) to yield a consistent ATE. Since OLS estimates are obtained by minimizing residual variance, this adjustment effectively removes selection bias due to imbalance in observables between treated and control groups.[^Idass]

[^Idass]: Keep in mind the key identifying assumption: \( E[\varepsilon_i | D_i, X_i] = E[\varepsilon_i | X_i] \) ensures that treatment assignment is independent of unobserved determinants of the outcome, given \( X_i \). OLS estimation of \( \delta \) follows the standard least squares formula \(\hat{\delta} = (Z'Z)^{-1} Z' Y\) where \( Z \) includes both the treatment variable \( D \) and the covariates \( X \). This adjustment accounts for selection bias due to observed covariates but does not address bias from unobserved confounders. The OLS estimate \( \hat{\delta} \) isolates the variation in \( Y \) that is orthogonal to \( X \), effectively controlling for selection on observables. The equivalence between OLS adjustment and residualization via the Frisch-Waugh-Lovell theorem is discussed in more detail at the end of this chapter.

If one wants to allow for genuine heterogeneity, an interacted model of the form 

\begin{equation} 
Y_i = \alpha + \delta\, D_i + X_i^\top \beta + (D_i \times X_i^\top)\gamma + \varepsilon_i 
\end{equation}

provides a richer description, where the difference between treatment and control depends on \( X_i \). The main ideas of averaging over subgroups remain, but the summation becomes slightly more involved. The conclusion, however, is identical: to recover an overall ATE, we compute the mean difference in fitted outcomes across the distribution of \( X_i \), which can be done analytically if the variables are centered, or by simply averaging predicted effects in the sample. More flexible or nonparametric estimators (e.g., random forests or series expansions) can approximate \(\delta^{CATE}(x)\) even when the relationship between \(X\) and outcomes is complex, but the principle remains the same: once \(\delta^{CATE}(x)\) is estimated for all relevant values of \(x\), you take the expectation of those conditional estimates over your data’s distribution of \(X\). This averaging yields the ATE, which is the sample-level effect of treatment.

Turning back to the scholarship example, once we consider different university types (\(X_i\)) as key observed covariates, we treat each subgroup (e.g., public, private, or non-profit) as though the scholarship were randomly allocated within that group. Estimating these subgroup-specific, or conditional, treatment effects and then weighting them by the share of each subgroup in the population yields the overall ATE. Alternatively, a single OLS regression including a dummy for scholarship receipt and controls for university type (plus other relevant variables) would provide a direct estimate of ATE under the assumption that within each observed subgroup, scholarship assignment is as good as random. If we wished to further explore heterogeneous effects, we would extend the model to include interactions, but the basic principle remains the same: regression-based adjustment in observational settings follows the same template as in randomized experiments, with the critical difference that we must now assume unobserved confounders do not invalidate our as-if-random assumption for each subgroup \(X = x\).

The link between conditional treatment effects (CATE) and the overall ATE is easiest to see when \( X \) is discrete, as in the scholarship example: we sum or weight each subgroup’s treatment effect \( \delta^{CATE}(x) \) by the proportion of units who fall into that subgroup. When \( X \) is continuous or high-dimensional, we instead integrate (or sum) the estimated effects over all observed \( X_i \) values.[^HighCat] In practice, you do not generally have to do an explicit integral. Regression models—particularly in their single-regression form—implement this averaging in one step, yielding a coefficient (\( \delta \) or an average of \( \delta + X_i^\top \gamma \)) that corresponds to the population-level effect of treatment.

[^HighCat]: When covariates are continuous or high-dimensional, the transition from the conditional average treatment effect (CATE) to the average treatment effect (ATE) follows the same core logic but replaces summation over discrete subgroups with integration (or summation) over the entire distribution of \(X\). Formally, the ATE in the presence of continuous (or multiple) covariates is:
\[
\text{ATE} \;=\; E[\delta^{CATE}(X)] \;=\; \int \delta^{CATE}(x) \, dF_X(x)
\]
where \(\delta^{CATE}(x) = E[Y(1) - Y(0)\mid X = x]\) and \(F_X(x)\) is the distribution of \(X\). If \(X\) has multiple continuous components (for example, age and years of education), then the integral is taken over their joint distribution. The principle behind regression-based methods is that one can estimate \(\delta^{CATE}(x)\) (or \(\delta\) in a simpler setting without interactions) for all \(x\) at once by fitting a model that relates \(Y\) to \(D\), \(X\), and possibly interactions, then averaging over the empirical distribution of \(X\).

**Inference and Robust Standard Errors**

Regardless of which regression specification is used, valid statistical inference depends on an appropriate variance estimator. In observational studies, heteroskedasticity is common, so robust (Eicker-Huber-White) standard errors are recommended. These account for potentially unequal error variances across units. Cluster-robust standard errors may be warranted if data exhibit intra-group correlation (e.g., if multiple observations share the same institution). Confidence intervals follow the usual construction:

\begin{equation}
\widehat{\delta} \pm z_{1-\alpha/2} \times \text{SE}(\widehat{\delta})
\end{equation}

where \( z_{1-\alpha/2} \) is from a standard normal or \( t \)-distribution.

Ultimately, the assumption that conditioning on \( X \) renders treatment assignment “as good as random” is critical. If unobserved confounders remain, none of these regression-based strategies will fully eliminate bias. Still, under selection on observables, the same frameworks that applied to RCTs can be deployed with only minor modifications—namely, stratifying or partialing out \( X \) to ensure that treated and untreated groups are comparable. Whether one uses separate regressions for treated and control groups or a single regression that pools all observations, the mechanics of going from CATE to ATE simply involve averaging predicted differences across the sample’s distribution of \( X \). If \( X \) is limited to a small set of categories, that average appears as a weighted sum of subgroup estimates; with continuous or high-dimensional covariates, the summation naturally extends to an integral or a regression-based average, providing a unified framework for causal inference in observational studies.

Beyond robust standard errors, the delta method provides an approximation for the standard error of the ATE when it is a nonlinear function of estimated regression coefficients. This nonlinearity can arise in several ways: (1) Nonlinear transformations of coefficients—for example, when ATE is expressed as a ratio of parameters, such as in elasticity estimation (\( \beta_1 / \beta_2 \)), requiring an approximation of its variance; (2) Treatment-covariate interactions—when interaction terms (\( D_i \times X_i^\top \gamma \)) are included, the marginal treatment effect varies with \( X_i \), making ATE a nonlinear function of coefficients; and (3) Quadratic or higher-order terms of confounders—if the regression includes squared terms (\( X_i^2 \)) or other nonlinear transformations, the standard error of ATE needs adjustment. While useful for analytical variance estimation, the accuracy of the delta method depends on the validity of a first-order Taylor expansion, and in some cases, bootstrapping may be more reliable. For a detailed discussion, see the end of the chapter. Bootstrapping provides another flexible approach to inference by resampling the data to approximate the sampling distribution of \(\delta\). This is particularly common when separate regressions are estimated for treated and control groups, allowing for nonparametric variance estimation. However, it can also be applied in a single regression framework by resampling observations with replacement and re-estimating the treatment effect across bootstrap samples as we covered in bootstrapping chapter. 

Before concluding this section with a simulation, we emphasize once again that the validity of causal treatment effect estimates hinges on key assumptions: conditional independence (no unobserved confounders once \( X \) is included), common support (positive probability of treatment at all \( X \) values), and correct model specification (adequately capturing nonlinearity and interactions). Researchers often test alternative specifications to ensure robustness. OLS provides consistent estimates under these assumptions, but its accuracy depends on correctly measuring and including all relevant covariates. Even small omissions can introduce omitted variable bias if excluded factors are correlated with both treatment and outcome. While regression adjustment remains a widely used and transparent approach, its reliability ultimately depends on the selection-on-observables assumption.

This simulation generates a synthetic dataset with a continuous outcome, covariates drawn from a multivariate normal distribution, and a binary treatment variable generated via a logistic model. The outcome is then constructed as a linear function of the treatment, covariates, and an error term. Various regression models are estimated to recover the average treatment effect (ATE). First, separate regressions are run for treated and control groups, and the ATE is computed as the difference in predicted outcomes at the mean covariate values of each group. Next, a single regression with adjustment for covariates is estimated using robust standard errors, and several interaction models are fitted—including models with treatment-by-covariate interactions and centered covariates—to assess heterogeneity and compare the consistency of the estimated treatment effect.


``` r
library(MASS)        # For multivariate normal distribution
library(lmtest)      # For hypothesis testing
library(boot)        # For bootstrapping
library(estimatr)    # For robust standard errors with lm_robust
set.seed(123) # Set seed for reproducibility
n <- 1000  # Sample size
p <- 3     # Number of covariates
# Generate covariates X ~ N(0, I)
X <- mvrnorm(n, mu = rep(0, p), Sigma = diag(p))
# Generate treatment assignment (D) from a logistic probability model
D_prob <- plogis(X %*% runif(p, -1, 1))  # Compute propensity scores
# D_prob <- plogis(X %*% c(0.5, -0.3, 0.2))  # Probability of treatment
D <- rbinom(n, 1, D_prob)  # Treatment indicator
# Generate outcome Y with treatment effect and confounders
gamma_true <- 3  # True treatment effect
beta <- runif(p, -1, 1)  # True coefficients for X
Y <- gamma_true * D + X %*% beta + rnorm(n)  # Outcome model
# Separate Regressions for Treated and Control Groups
treated_model <- lm(Y ~ X, subset = (D == 1))  # Regression for treated group
control_model <- lm(Y ~ X, subset = (D == 0))  # Regression for control group
X_treated_mean <- colMeans(X[D == 1,])  # Mean covariates for treated group
X_control_mean <- colMeans(X[D == 0,])  # Mean covariates for control group
#fitted potential outcomes
Y1_hat <- predict(treated_model, newdata = data.frame(t(X_treated_mean))) 
Y0_hat <- predict(control_model, newdata = data.frame(t(X_control_mean))) 
# Compute ATE using the average of the difference of pot.outcome of units
ATE_separate <- mean(Y1_hat - Y0_hat)
ATE_separate
```

```
## [1] 2.990066
```

``` r
# Single OLS Regression with Adjustment for X (Robust SE HC2)
ols_robust_hc2 <- lm_robust(Y ~ D + X, se_type = "HC2")  # Robust SEs (HC2)
ATE_robust <- coef(ols_robust_hc2)["D"]
ATE_robust
```

```
##        D 
## 2.989047
```

``` r
coef(summary(ols_robust_hc2))[2, "Std. Error"]  # Standard error for D in OLS model
```

```
## [1] 0.06491384
```

``` r
# Interaction Model: D * X (Robust SE HC2)
interaction_model <- lm_robust(Y ~ D + X + D:X, se_type = "HC2")  # Interaction model
ATE_interaction <- coef(interaction_model)["D"]
ATE_interaction
```

```
##        D 
## 2.993935
```

``` r
coef(summary(interaction_model))[2, "Std. Error"]  # Standard error for D in OLS model
```

```
## [1] 0.06502589
```

``` r
# Interaction Model: D * (X - X_mean) (Centered Interaction Model)
X_mean <- colMeans(X)  # Compute mean of each covariate
X_centered <- sweep(X, 2, X_mean, "-")  # Center X
interaction_model_centered<-lm_robust(Y ~ D + X_centered + D:X_centered, 
                                      se_type = "HC2")
ATE_interaction_centered <- coef(interaction_model_centered)["D"]
ATE_interaction_centered
```

```
##        D 
## 2.990066
```

``` r
coef(summary(interaction_model_centered))[2, "Std. Error"] 
```

```
## [1] 0.06488983
```

``` r
# Higher-Order Interaction Model: X^2 + X^3 + Cross-product-terms
X_sq <- X^2  # Squared terms
X_cube <- X^3  # Cubed terms
X_cross <- X[,1] * X[,2] + X[,1] * X[,3] + X[,2] * X[,3]  # Cross-product terms 
higher_order_model <- lm_robust(Y ~ D + X + X_sq + X_cube + 
                            D:X + D:X_sq + D:X_cube + X_cross, se_type = "HC2")
ATE_higher_order <- coef(higher_order_model)["D"]
ATE_higher_order
```

```
##        D 
## 2.981863
```

``` r
coef(summary(higher_order_model))[2, "Std. Error"] 
```

```
## [1] 0.1031995
```

Although the simulation uses generated data, the same steps can be applied to any dataset by uploading it instead. As demonstrated, the ATE from separate regressions (computed as the difference in expected outcomes) closely aligns with that from the centered interaction model, echoing results from the randomization chapter in our previous work. In practice, the treatment effect coefficient (D) from a simple regression is typically similar, both in point estimates and confidence intervals, to those obtained from more complex interacted models, which is why many studies only report the single-regression (D) model. However, when higher-order terms and numerous interactions are added—potentially increasing the number of predictors beyond the sample size—techniques like double lasso become crucial for proper variable selection and inference, a topic we will explore next.

To address cases where \( X \) is high-dimensional or difficult to specify, we next introduce Double Machine Learning methods, which remains a linear approach but selects relevant covariates efficiently. 


