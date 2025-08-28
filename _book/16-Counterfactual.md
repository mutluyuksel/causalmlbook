# Counterfactual Framework

Causal inference lies at the heart of understanding the effects of interventions, policies, or treatments. The term "causal inference" is sometimes misused in informal contexts to mean identifying variables/features associated with an outcome rather than establishing "true" cause-and-effect relationships. This confusion appears in some blogs, online tutorials, and discussions in data science communities, where they refer feature importance or explanatory variables as evidence of causality. As discussed in detail in Chapter 2, one of the key motivations for causal inference is the fundamental distinction between correlation and causation. While correlation can identify patterns and associations, causation establishes whether one variable influences another. This distinction is essential for guiding decisions in fields like economics, health, and social sciences, where policymakers and practitioners require actionable insights.

It is important to clarify key conceptual issues before we start the causal inference chapters. While we may observe differences in outcomes between two groups, we cannot directly attribute these differences to group-specific characteristics alone, as confounding factors, which lead to omitted variable bias, selection bias, or unobserved variables, may be driving the observed differences. Also, we cannot say for sure that a specific characteristic caused those outcome differences directly. Simply observing differences or identifying an association does not establish causality; for a characteristic, event, or variable to be considered causal, it must be possible to manipulate, treat, or intervene in it. For example, as discussed in Simpson’s paradox in Chapter 2, we cannot conclude that being older directly causes lower income because age itself cannot be manipulated. The same applies to characteristics such as gender or race—observed differences in outcomes may result from structural factors, confounding variables, or other unobservable variables rather than the characteristic itself. However, if we introduce an intervention—such as providing older workers with special security training or equipment—and observe that they can work night shifts more often, leading to higher earnings, then we can say that the special training or equipment caused the increase in income among older workers. The key distinction is that the treatment (training or equipment) is manipulable, allowing us to assess its causal effect on the outcome. This is the essence of causal inference: understanding the effect of an intervention, policy or treatment on an outcome by comparing the observed outcome with the outcome that would have been observed under a different treatment. Causal inference relies on several assumptions, and in these chapters, we will cover both the methods and the assumptions required for valid causal analysis.

In policymaking and business, understanding causal relationships is critical. Decisions, such as whether increasing education funding improves student outcomes or raising the minimum wage reduces poverty, hinge on knowing the direction and magnitude of these effects. In healthcare, policymakers must determine whether expanding access to preventive care reduces long-term hospitalizations or whether a new drug truly improves patient recovery rates rather than just correlating with better outcomes. Similarly, firms must evaluate whether advertising campaigns boost sales or if seasonal trends are the real cause. Causal inference provides the rigorous framework needed to evaluate such policies, going beyond mere observation of statistical associations to uncover real cause-and-effect relationships.

Traditional machine learning methods, which are typically optimized for prediction, are not inherently designed to answer causal questions. Predictive models may identify spurious correlations or overlook confounding variables, leading to misleading inferences about cause and effect. To overcome these limitations, causal ML techniques—such as double/debiased machine learning, causal forests, and methods that integrate causal graphs—have been developed. These approaches bridge the gap between prediction and causation by enabling the estimation of causal effects while leveraging ML’s strengths in handling high-dimensional, complex data.

For readers unfamiliar with the foundational concepts of causal inference, we recommend reviewing Chapter 2, where these topics, including the distinction between correlation, regression, and the causal effect, are explored in depth. This chapter will build on those concepts, focusing on methods and their practical applications. It will cover the counterfactual framework, including the definition of counterfactuals, the challenges of missing data and state, and how machine learning methods can estimate counterfactuals under identification assumptions. The chapter will also explore the fundamental problem of causal inference by examining the role of randomization and the interplay between experimental and observational data. Randomized controlled trials, cluster randomization, and adaptive experimental designs optimized through machine learning will also be discussed, as will partial effects, focusing on marginal effects in linear and nonlinear models and their estimation through machine learning-based predictions. <!-- this sentence will be removed if cannot mention this in RCT section -->

Selection on observables will then be examined in detail, covering matching methods, weighting strategies, and other approaches such as doubly robust. These methods address challenges posed by confounding factors when causal inference relies on observational data. Finally, the next chapter will focus on selection on unobservables, explore techniques like instrumental variables, difference-in-differences, regression discontinuity designs, and synthetic control methods. One chapter later, we will discuss heterogeneous causal methods such as double machine learning, causal forests, and related approaches. Together, these chapters provide a comprehensive understanding of causal inference techniques based on potential outcome framework and their integration with machine learning.

For readers seeking a deeper understanding of causal inference methods in economics, we recommend widely regarded texts such as Causal Inference: The Mixtape by Scott Cunningham, The Effect by Nick Huntington-Klein, and Mostly Harmless Econometrics by Angrist and Pischke, A First Course in Causal Inference by Peng Ding. These books provide excellent coverage of foundational concepts and techniques, from randomized controlled trials to instrumental variables, regression discontinuity, and difference-in-differences. While the purpose of these chapters is to offer a comprehensive overview of these methods, the primary focus is to demonstrate how machine learning can complement and improve traditional approaches. By providing intuitive explanations and practical insights, we aim to show how ML methods can improve causal inference in areas like high-dimensional data, flexible modeling, and estimating heterogeneous treatment effects. Let's start our discussion with the counterfactual framework.

## Counterfactual Framework

Imagine studying the "causal" effect of obtaining a college degree on earnings. For an individual with a degree, we observe their income \(Y_i(1)\) but not the counterfactual income \(Y_i(0)\) they would have earned without the college degree; similarly, for someone without a degree, we observe their income \(Y_i(0)\) but not the income \(Y_i(1)\) they might have earned with a degree. In theory, the "causal" effect for each person is \(Y_i(1) - Y_i(0)\), but since both states cannot be observed simultaneously, individual causal effects remain unmeasurable. Now, consider a population of \(N\) individuals—say, a quarter have a degree—and imagine a parallel universe where each person’s potential outcomes (income with a degree in one universe/state; income without a degree in another universe/state) are observable. In that case, the causal effect of a college degree for each person could be determined by calculating the difference between their incomes in the actual and counterfactual states, \(Y_i(1) - Y_i(0)\). By aggregating these individual causal effects across the entire population, we could compute the Average Treatment/Causal Effect of treatment (college degree, here) on outcome (income, here) as the mean of these differences, \(\mathbb{E}[Y_{i}(1) - Y_{i}(0)]\). In this scenario, the Average Treatment Effect (ATE) can also be computed as \(\mathbb{E}[Y_i(1)] - \mathbb{E}[Y_i(0)]\), which is equivalent to the difference between the average income if everyone had a degree and if no one did. This framework precisely defines the population-level average treatment/causal effect of education on earnings. However, there are other causal effect measures, which we will cover in later sections.

Because of the inability to observe the outcome in both the treated state and the control state for the same individual, as well as the confounding factors that complicate causal interpretation, Donald Rubin and others developed the Rubin Causal Model (RCM) (Rubin, 1974; Holland, 1986). The RCM provides a formal framework for defining and estimating causal effects in the presence of the fundamental problem of causal inference, which is defined as "for each unit, only one of the two potential outcomes—\(Y_i(1)\) (if treated) or \(Y_i(0)\) (if untreated)—can be observed, depending on the treatment assignment. The other outcome remains unobservable, referred to as the counterfactual, creating a missing data problem. The potential outcomes framework is also known as the "counterfactual framework".[^countEx] As a result, direct calculation of the causal effects at the individual or population level is impossible. This challenge forces researchers to rely on assumptions and robust designs to approximate the unobservable counterfactuals and isolate causal effects.

[^countEx]: Counterfactual scenarios are a popular theme in literature, film, and TV, offering thought-provoking explorations of alternate realities. The movie *Sliding Doors* examines how a single missed train transforms a woman's life, while Robert Frost's poem *The Road Not Taken* poetically reflects on the choices that shape our paths. Similarly, *The Man in the High Castle* envisions a world where the Axis powers won World War II, and Philip Roth's *The Plot Against America* reimagines U.S. history under a Lindbergh presidency. Documentaries like *The Fog of War* analyze pivotal decisions during conflicts, illustrating how alternative choices might have changed history. These examples mirror the counterfactual frameworks in causal inference, emphasizing how different "what if" scenarios reveal the impact of specific interventions or decisions.

A key concept in causal inference is that the unit of analysis may differ from the physical unit being studied. Consider the effect of a college degree on income: if I initially earned a low salary without a degree and later earned more after obtaining one, it might seem that we observe both potential outcomes—\(Y_i(0)\) (without a degree) and \(Y_i(1)\) (with a degree)—allowing us to estimate the causal effect. However, this reasoning is flawed because my "before" and "after" measurements represent two distinct units, meaning we have four potential outcomes—\(Y_{i,\text{before}}(0)\), \(Y_{i,\text{before}}(1)\), \(Y_{i,\text{after}}(0)\), and \(Y_{i,\text{after}}(1)\)—with only two observed, and two missing. For instance, if my income increased over time without obtaining a degree, one might erroneously conclude that a college degree has no effect; similarly, if my income decreased after earning a degree, the effect might be falsely deemed negative. This example highlights the need to clearly define the unit in causal analysis. Causal inference requires multiple units exposed to different treatments, either by observing the same unit over time or different units simultaneously. However, having multiple units alone does not resolve the fundamental problem of causal inference.

The RCM assumes that each unit has a well-defined treatment status (\(D_i\)) and potential outcomes (\(Y_i(1)\) and \(Y_i(0)\)). The observed outcome (\(Y_i\)) depends on the treatment received, given by:  

\begin{equation}
Y_i = D_i \cdot Y_i(1) + (1-D_i) \cdot Y_i(0)
\label{eq:potential_outcome}
\end{equation}

This equation states that if \(D_i = 1\), the observed outcome is the potential outcome in the treatment state (\(Y_i = Y_i(1)\)), and if \(D_i = 0\), it is the potential outcome in the control state (\(Y_i = Y_i(0)\)). If both potential outcomes were observable, identifying individual treatment effects (\(Y_i(1) - Y_i(0)\)) and computing the population-level average treatment effect (\(\mathbb{E}[Y_i(1) - Y_i(0)]\)) would be straightforward. 

<!--\begin{table}[h]
    \centering
    \renewcommand{\arraystretch}{1.5} % Increases row spacing
    \caption{Potential Outcomes and Individual Treatment Effects} % Table title
    \label{tab:potential_outcomes} % Table label for referencing
    \vspace{0.5cm} % Adds space before the table
    \resizebox{1\textwidth}{!}{%
    \begin{tabular}{|c|c|c|c|c|}
        \hline
        $D$ & $Y(1)$ Potential Outcome  & $Y(0)$ Potential Outcome  & Individual Treatment  & $Y$ Observed  \\
         & Treatment state&  NO Treatment(Control) state & Effect & Outcome \\
        \hline
        1 & $Y_T(1) \text{ observed}$ & $Y_T(0) \text{ what-if}$ & $Y_T(1) - Y_T(0)$ & $Y_T$ \\
        \hline
        1 & $Y_T(1) \text{ observed}$ & $Y_T(0) \text{ what-if}$ & $Y_T(1) - Y_T(0)$ & $Y_T$ \\
        \hline
        1 & $Y_T(1) \text{ observed}$ & $Y_T(0) \text{ what-if}$ & $Y_T(1) - Y_T(0)$ & $Y_T$ \\
        \hline
        0 & $Y_C(1) \text{ what-if}$ & $Y_C(0) \text{ observed}$ & $Y_C(1) - Y_C(0)$ & $Y_C$ \\
        \hline
        0 & $Y_C(1) \text{ what-if}$ & $Y_C(0) \text{ observed}$ & $Y_C(1) - Y_C(0)$ & $Y_C$ \\
        \hline
        0 & $Y_C(1) \text{ what-if}$ & $Y_C(0) \text{ observed}$ & $Y_C(1) - Y_C(0)$ & $Y_C$ \\
        \hline
    \end{tabular}%
    }
\end{table}-->


Table: (\#tab:unnamed-chunk-1)Potential Outcomes and Individual Treatment Effects

| $D$|<span style='font-weight:normal;'>$Y(1)$ <br> Potential Outcome<br>Treatment state</span> |<span style='font-weight:normal;'>$Y(0)$ <br> Potential Outcome<br>No Treatment state</span> |<span style='font-weight:normal;'>Individual <br> Treatment <br> Effect</span> |<span style='font-weight:normal;'>$Y$ <br> Observed Outcome</span> |
|---:|:-----------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------|:------------------------------------------------------------------|
|   1|$Y_{T}(1)$ observed                                                                       |$Y_{T}(0)$ what-if                                                                           |$Y_{T}(1) - Y_{T}(0)$                                                          |$Y_{T}$                                                            |
|   1|$Y_{T}(1)$ observed                                                                       |$Y_{T}(0)$ what-if                                                                           |$Y_{T}(1) - Y_{T}(0)$                                                          |$Y_{T}$                                                            |
|   1|$Y_{T}(1)$ observed                                                                       |$Y_{T}(0)$ what-if                                                                           |$Y_{T}(1) - Y_{T}(0)$                                                          |$Y_{T}$                                                            |
|   0|$Y_{C}(1)$ what-if                                                                        |$Y_{C}(0)$ observed                                                                          |$Y_{C}(1) - Y_{C}(0)$                                                          |$Y_{C}$                                                            |
|   0|$Y_{C}(1)$ what-if                                                                        |$Y_{C}(0)$ observed                                                                          |$Y_{C}(1) - Y_{C}(0)$                                                          |$Y_{C}$                                                            |
|   0|$Y_{C}(1)$ what-if                                                                        |$Y_{C}(0)$ observed                                                                          |$Y_{C}(1) - Y_{C}(0)$                                                          |$Y_{C}$                                                            |


This table illustrates the notations we discussed above, helping us understand the estimation of treatment effects when only one outcome per individual is observed. Each unit (student) has two potential outcomes: one if they receive the scholarship (\(Y(1)\)) and one if they do not (\(Y(0)\)). The table separates individuals into two groups: those who receive the scholarship (\(D = 1\)) and those who do not (\(D = 0\)). For treated students, the observed outcome is \(Y_T(1)\) (earnings after receiving the scholarship, while the counterfactual outcome \(Y_T(0)\) (what their earnings would have been without the scholarship; labeled "what-if")) is unobserved. In contrast, for the control group, we observe \(Y_C(0)\) (earnings without the scholarship) and do not see \(Y_C(1)\) (what their earnings would have been if they had received the scholarship; labeled "what-if")). The individual treatment effect is defined as the difference \(Y(1) - Y(0)\). If both outcomes were observable for every student, averaging the differences would yield the Average Treatment Effect (ATE). However, because one outcome is always missing, we must estimate the ATE using observed data and statistical methods. This table therefore highlights the distinction between observed outcomes and unobserved counterfactuals, which is key for causal inference.

<!--\begin{table}[h]
    \centering
    \renewcommand{\arraystretch}{1.5} % Increases row spacing
    \caption{Expected Potential Outcomes and Treatment Effects} % Table title
    \label{tab:expected_outcomes} % Table label for referencing
    \vspace{0.5cm} % Adds space before the table
    \resizebox{1\textwidth}{!}{%
    \begin{tabular}{|l|l|l|l|l|}
        \hline
        & \(E[Y(1)]\) Expected Potential  & \(E[Y(0)]\) Expected Potential  & Difference & \(E[Y]\) \\
        & Outcome at Treatment state & Outcome at Control state &  &  \\
        \hline
        \(\mathbf{D=1}\) & \(E[Y(1)|D=1]\) (Observable for Treated) & \(E[Y(0)|D=1]\) (Unobservable for Treated) & ATT & \(E[Y|D=1]\) \\
        \(\mathbf{D=0}\) & \(E[Y(1)|D=0]\) (Unobservable for Control) & \(E[Y(0)|D=0]\) (Observable for Control) & ATU & \(E[Y|D=0]\) \\
        \(\mathbf{Sample}\) & \(E[Y(1)]\) & \(E[Y(0)]\) & ATE &  \\
        &  & Selection Bias: \(E[Y(0)|D=1] - E[Y(0)|D=0]\) &  & Simple Diff. \\
        \hline
    \end{tabular}%
    }
\end{table}-->


Table: (\#tab:unnamed-chunk-2)Expected Potential Outcomes and Treatment Effects

|       |<span style='font-weight:normal;'>$E[Y(1)]$ <br> Expected Potential Outcome at<br>Treatment state</span> |<span style='font-weight:normal;'>$E[Y(0)]$ <br> Expected Potential Outcome at<br>Control state</span> |<span style='font-weight:normal;'>Difference</span> |<span style='font-weight:normal;'>$E[Y]$</span>                                     |
|:------|:--------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------|:---------------------------------------------------|:-----------------------------------------------------------------------------------|
|$D=1$  |$E[Y(1)\mid D=1]$ (Observable for Treated)                                                               |$E[Y(0)\mid D=1]$ (Unobservable for Treated)                                                           |ATT                                                 |$E[Y\mid D=1]$                                                                      |
|$D=0$  |$E[Y(1) \mid D=0]$ (Unobservable for Control)                                                            |$E[Y(0) \mid D=0]$ (Observable for Control)                                                            |ATU                                                 |$E[Y \mid D=0]$                                                                     |
|Sample |$E[Y(1)]$                                                                                                |$E[Y(0)]$                                                                                              |ATE                                                 |                                                                                    |
|       |                                                                                                         |Selection Bias: <span style='font-size:0.5em;'>$E[Y(0)\mid D=1] - E[Y(0)\mid D=0]$</span>              |                                                    |Simple Differ.:<span style='font-size:0.5em;'>$E[Y\mid D=1] - E[Y \mid D=0]$</span> |

The table above links our data to the Rubin Causal Model by showing how we compute expected potential outcomes. For example, to calculate \(E[Y(1)|D=1]\), we take the observed outcomes for the treated group—the column showing \(Y(1)\) for students who received the scholarship (\(D=1\)s)—and compute their average. This average represents the expected outcome under treatment for the treated units. Similarly, for the control group, we average the observed outcomes in the \(Y(0)\) column to obtain \(E[Y(0)|D=0]\), which is the expected outcome under no treatment for these units.

The table also shows how these group-specific averages contribute to overall expectations. For instance, the overall expected potential outcome under treatment, \(E[Y(1)]\), is a weighted combination of \(E[Y(1)|D=1]\) and \(E[Y(1)|D=0]\), even though for the control group the outcome \(Y(1)\) is unobserved (thus, needs to be estimated). The same idea applies to \(E[Y(0)]\) using \(E[Y(0)|D=1]\) (unobserved for the treated) and \(E[Y(0)|D=0]\) (observed for the control).

This process of averaging helps us understand how we move from individual data points to group-level expectations and, ultimately, to estimates of causal effects. The table also highlights the difference in expectations, such as the simple difference between \(E[Y|D=1]\) and \(E[Y|D=0]\), which provides a preliminary estimate of the treatment effect. However, since we only observe one potential outcome per student, unobserved counterfactuals can lead to selection bias. For instance, the difference \(E[Y(0)|D=1] - E[Y(0)|D=0]\) represents how the unobserved outcomes among the treated differ from those in the control group.

This visual framework is essential as we move forward. Next, we will discuss the assumptions that let us approximate these missing counterfactuals, formally define treatment effects such as ATE, ATT, and ATU, and address selection bias. Later in the chapter, we will also cover methods like randomization, regression, matching, and weighting, all of which rely on this framework to estimate causal effects accurately.

### Assumptions of the Rubin Causal Model:

The Stable Unit Treatment Value Assumption (SUTVA) is essential for defining potential outcomes (Rubin, 1980). SUTVA consists of two key components: no interference and consistency. The no-interference assumption states that the potential outcomes of observation \(i\) are independent of the treatment assignments of all other units. In other words, a unit’s potential outcomes are not affected by spillover or network effects from the treatment of others. This assumption rules out general equilibrium effects, which can be problematic in economic settings where policy interventions influence neighboring households or firms. When a treated group represents a large share of the population, assuming no spillovers may not be plausible. Consequently, policy recommendations based on randomized or natural experiments should be reassessed for broader populations, as small-sample studies may satisfy SUTVA while large-scale implementations violate it. Violations, such as spillovers or treatment diffusion, require alternative approaches, including network-based models and machine learning techniques for spillover detection.  

The consistency assumption states that the observed outcome for a unit under its actual treatment status equals the corresponding potential outcome: \(Y_i = D_i \cdot Y_i(1) + (1-D_i) \cdot Y_i(0)\). This ensures that the treatment received aligns exactly with the treatment assigned, ruling out hidden variations in treatment (e.g., different treatment intensities, misreporting, or noncompliance). If multiple versions of a treatment exist but are not accounted for, causal estimates may be biased. In practice, inconsistencies arise due to implementation errors, partial treatment uptake, or measurement issues. Machine learning methods, such as anomaly detection and data imputation, can help identify and mitigate such inconsistencies, improving the robustness of causal analysis.

In the potential outcomes framework, causal identification relies on the ignorability (or unconfoundedness/conditional independence) assumption, i.e., \((Y_i(1), Y_i(0)) \perp D_i \mid X_i\), which asserts that treatment assignment is as good as random once conditioned on covariates \(X_i\). In another words, an assignment is unconfounded if the assignment mechanism does not depend on the potential
outcomes and depend only on the pre-treatment covariates. A perfect randomized controlled trial would eliminate the need for this assumption. However, in observational studies, this assumption is required to employ statistical adjustments—such as matching, weighting, and regression—to mimic randomization. Machine learning further improves these methods by improving propensity score and outcome model estimation. In high-dimensional settings, ML-based variable selection (e.g., LASSO, random forests) effectively isolates relevant confounders, mitigating biases from model misspecification or omitted variables and bolstering the credibility of causal estimates when true randomization is not feasible.

With these assumptions in place, we now turn to the most common linear causal estimand (target quantity, which is function of potential outcomes) in the RCM framework.

## Average Treatment Effects:

Even though we cannot observe the treatment effect for each individual, the collection of individual causal effects within a population forms a distribution. This distribution allows us to estimate key parameters such as the mean (Average Treatment Effect, ATE), variance, and other descriptive statistics.[^AteWork]

[^AteWork]: A distribution is simply a collection of data or scores on a variable, typically arranged from smallest to largest and often represented graphically. — Page 6, *Statistics in Plain English*, Third Edition, 2010.  

The Average Treatment Effect (ATE) is the most common parameter of interest (i.e. estimand) in causal inference within the RCM framework. It represents the average of individual treatment effects, \(Y_i(1) - Y_i(0)\), serving as a crucial measure in policy evaluation, program assessment, and medical research. Formally, it is defined as: 

\begin{equation}
\delta^{ATE} = \mathbb{E}[\delta_i] = \mathbb{E}[Y_i(1) - Y_i(0)] = \mathbb{E}[Y_i(1)] - \mathbb{E}[Y_i(0)]
\label{eq:ATE}
\end{equation}

To understand this intuitively, consider a government implementing a policy, an agency introducing a new regulation, or a doctor prescribing a medication. In each scenario, we can conceptualize two states: a treatment state where all units in the population are exposed to the intervention, and a control state where none are exposed. The equation \(\mathbb{E}[Y_i(1) - Y_i(0)]\) represents the population average of the individual treatment effects, while \(\mathbb{E}[Y_i(1)] - \mathbb{E}[Y_i(0)]\) captures the difference in the expected outcomes for the entire population if all units were treated versus untreated. Under the SUTVA, which ensures no interference between units and consistent treatment effects, these two expressions are equivalent.

However, like the individual-level treatment effect, the ATE is inherently unknowable because it requires knowledge of both potential outcomes, \(Y_i(1)\) and \(Y_i(0)\), for each unit. Due to the fundamental problem of causal inference, only one of these outcomes is observable for any given unit. As a result, the ATE is not a quantity that can be directly calculated but rather one that can only be estimated with different methods (i.e. estimators) under certain statistical assumptions.[^assump1] Each method in the RCM framework aims to estimate the ATE by leveraging different assumptions and estimators to address missing counterfactuals and isolate causal effects. In the following sections, we will explore these estimation techniques in detail.

[^assump1]: Keep in my mind estimator is functions (statistics) of observed data, usually denote by a hat. For the same estimand, there can be multiple estimators. Desirable properties of estimators are Unbiased/Consistency and Efficiency (low variance).

Depending on the research question, we may also be interested in estimating other treatment effects beyond the ATE. The Average Treatment Effect on the Treated (ATT) measures the average treatment effect specifically for the subgroup of individuals who actually received the treatment. It is formally defined as:  

\begin{equation}
ATT = \mathbb{E}[Y_{i}(1) \mid D=1] - \mathbb{E}[Y_{i}(0) \mid D=1]
\label{eq:ATT}
\end{equation}

In observational data, the ATT almost always differs from the ATE because individuals tend to sort into treatment based on expected gains. For instance, individuals with higher anticipated benefits from a policy or intervention may be more likely to participate, introducing endogenous selection into the treatment group. Like the ATE, the ATT is unknowable because it requires knowledge of both potential outcomes for each treated individual, which is impossible to observe directly; thus, it must be estimated using causal inference methods.

The Average Treatment Effect on the Untreated (ATU) measures the average treatment effect for those who did not receive the treatment (the control or untreated group). It is defined as:  

\begin{equation}
ATU = \mathbb{E}[Y_{i}(1) \mid D=0] - \mathbb{E}[Y_{i}(0) \mid D=0]
\label{eq:ATU}
\end{equation} 

As with the ATT, the ATU is also unknowable in observational data, as it requires observing both potential outcomes for individuals in the control group. Furthermore, if treatment effects are heterogeneous, the ATU will differ from the ATT, reflecting the varying impacts of treatment across different subpopulations.

Beyond the ATE, ATT, and ATU, additional parameters are used to estimate causal effects in specific contexts or with particular methods. These parameters such as CATE (Conditional Average Treatment Effect), LATE (Local Average Treatment Effect), and heteregeneous treatment effects will be explored in greater detail in the next chapter as we discuss approaches to causal inference tailored to different assumptions and data structures. 

Binary outcomes, common in medical and health studies, require specialized causal estimands due to their nature. In this context, we define \(\mu_1 = \text{Pr}(Y_i(1) = 1)\) and \(\mu_0 = \text{Pr}(Y_i(0) = 1)\), representing the probabilities of an outcome under treatment and control, respectively, often referred to as causal risks in medical research. The three primary causal estimands for binary outcomes are the causal risk difference (\(\delta_{RD} = \mu_1 - \mu_0\), equivalent to the ATE), the causal risk ratio (\(\delta_{RR} = \mu_1 / \mu_0\)), and the causal odds ratio (\(\delta_{OR} = \frac{\mu_1 / (1-\mu_1)}{\mu_0 / (1-\mu_0)}\)). It is important to note that the ratio estimands (e.g., risk ratio and odds ratio) are not averages of individual causal effects unless the effects are homogeneous across individuals. These estimands provide nuanced insights into treatment effects for binary outcomes, extending beyond the simple ATE to address specific contexts, especially in medical and epidemiological research.

Before we continue, we want to highlight an important causal estimands more common in medical and health studies. Binary outcomes require specialized causal estimands due to their nature. In this context, we define \(\mu_1 = \text{Pr}(Y_i(1) = 1)\) and \(\mu_0 = \text{Pr}(Y_i(0) = 1)\), representing the probabilities of an outcome under treatment and control, respectively, often referred to as causal risks in medical research. The three primary causal estimands for binary outcomes are the causal risk difference (\(\delta_{RD} = \mu_1 - \mu_0\), equivalent to the ATE), the causal risk ratio (\(\delta_{RR} = \mu_1 / \mu_0\)), and the causal odds ratio (\(\delta_{OR} = \frac{\mu_1 / (1-\mu_1)}{\mu_0 / (1-\mu_0)}\)). It is important to note that the ratio estimands (e.g., risk ratio and odds ratio) are not averages of individual causal effects unless the effects are homogeneous across individuals. These estimands provide nuanced insights into treatment effects for binary outcomes, extending beyond the simple ATE to address specific contexts, especially in medical and epidemiological research.

## Selection Bias and Heterogeneous Treatment Effect Bias

The fundamental problem of causal inference prevents the direct calculation of causal effects, so researchers often rely on survey or administrative data to estimate them in economics, health, and social sciences. A common but flawed approach is interpreting raw differences in average outcomes between groups (e.g., treated versus untreated) as the Average Treatment Effect (ATE) of the treatment, intervention, or policy. This approach fails to account for confounding factors—factors that influence both treatment and outcome—and overlooks selection bias, arising from non-random assignment to the treatment group, as well as heterogeneous effects, caused by the differential impact of treatment on treated and control groups.

Selection bias arises when the process determining treatment assignment is correlated with potential outcomes, leading to systematic differences between groups. For example, in observational studies, treatment is often not randomly assigned, and pre-existing differences between treated and untreated groups influence both the probability of receiving treatment and outcomes. As a result, the simple difference in average outcomes captures not just the treatment effect but also these underlying differences, conflating the two and biasing estimates. 

Our last table breaks down the observed and unobserved components of potential outcomes, helping us see how we calculate group averages and, ultimately, causal effects. For example, the column labeled \(E[Y(1)|D=1]\) represents the average outcome for students who received the scholarship. We compute this by taking each treated student's observed earnings and then averaging these values. Similarly, \(E[Y(0)|D=0]\) is the average outcome for students who did not receive the scholarship, calculated by averaging their observed earnings.

When we compute the simple difference \(E[Y|D=1] - E[Y|D=0]\) (as shown in the “Simple Diff.” column), we are subtracting the average earnings of the control group from the treated group. However, this difference does not purely reflect the causal effect of the scholarship. Instead, it contains two components: the true treatment effect on the treated (ATT) and a term capturing selection bias.

In our lasy table, the Average Treatment Effect on the Treated (ATT) focuses solely on the treated group. For students who receive the scholarship, the table shows their observed outcome under treatment—labeled as \(E[Y(1)|D=1]\)—and it also reminds us that their potential outcome if they had not received the scholarship, \(E[Y(0)|D=1]\), is unobserved. The ATT captures the difference between these two values: it tells us how much, on average, the scholarship changes the earnings for the treated students compared to what their earnings would have been without the scholarship. 

We can express the simple difference by adding and subtracting the unobserved counterfactual average \(E[Y(0)|D=1]\) from the treated group:

\begin{align}
\underbrace{\mathbb{E}[Y(1)\mid D=1] - \mathbb{E}[Y(0)\mid D=0]}_{\text{Simple Difference in average Outcomes (SDO)}} 
&= \underbrace{\left(\mathbb{E}[Y(1)\mid D=1] - \mathbb{E}[Y(0)\mid D=1]\right)}_{\text{Average Treatment Effect on the Treated (ATT)}} \notag \\
&\quad + \underbrace{\left(\mathbb{E}[Y(0)\mid D=1] - \mathbb{E}[Y(0)\mid D=0]\right)}_{\text{Selection Bias}}
\end{align}


Here, the first term \(\{E[Y(1)|D=1]-E[Y(0)|D=1]\}\) is the ATT—the true causal effect of receiving the scholarship for those who did. The second term \(\{E[Y(0)|D=1]-E[Y(0)|D=0]\}\) is the selection bias, indicating that if treated students would have had different earnings than control students even without the scholarship, then part of the observed difference is due to these pre-existing differences rather than the treatment itself.

For instance, if students who receive scholarships tend to come from backgrounds that predispose them to higher earnings (reflected in a higher \(E[Y(0)|D=1]\) compared to \(E[Y(0)|D=0]\)), then the selection bias term becomes positive. As a result, simply comparing average earnings would overstate the true effect of the scholarship. Similarly, in a health intervention study, individuals who choose to participate in a wellness program may already be more health-conscious and predisposed to better outcomes than those who opt out, regardless of the treatment itself. In both cases, observed differences in outcomes conflate the true causal effect with the effects of selection bias, making it impossible to interpret the results as causal without addressing this issue.

In some cases, we can also relate the simple difference to the Average Treatment Effect (ATE) by a similar reasoning, leading to a decomposition where the naive difference equals the ATE plus a selection bias term that accounts for differences in potential outcomes between treated and control units.

The goal of causal inference methods is to minimize or eliminate this bias by imposing identifying assumptions and using statistical techniques to adjust for these systematic differences. In the following chapters, we will explore these methods in detail, focusing on their ability to approximate the counterfactual outcomes for valid causal interpretation.

Heterogeneous Treatment Effect Bias arises because treatments often affect individuals or groups differently. For example, a policy intervention may benefit some subgroups more than others, resulting in varying treatment effects across the population. The difference between the Average Treatment Effect on the Treated (ATT) and the Average Treatment Effect on the Untreated (ATU) reflects this heterogeneity. When multiplied by the proportion of the control group, \((1-\pi)(ATT - ATU)\), it quantifies the bias introduced by assuming homogeneous treatment effects. 

To simplify analysis, researchers often assume constant treatment effects (i.e., no heterogeneity), which eliminates this bias. Even though this is a strong assumption, this is very common and plausible in social sciences and economics as we want to analyze average effects, not individual effect. That average treatment/causal effect is presented either treatment effect for average person or "homogeneous" average treatment effect for everyone. However, recent advances in machine learning allow researchers to better capture and model heterogeneous treatment effects, enabling more granular and targeted policy insights.

As we discussed above, the simple difference between the average outcomes of treated and untreated groups is often misinterpreted as the causal effect of treatment. This interpretation holds true only if selection bias and heterogeneous treatment effect bias are absent, which is rarely the case in practice. Addressing selection and heterogeneous treatment effect biases requires robust causal inference methods that impose appropriate identifying assumptions, helping to isolate the true treatment effect and ensure accurate estimates.

This simple difference in outcomes between treated (\(D=1\)) and untreated groups (\(D=0\)) can be decomposed ATE, selection bias, and heterogeneous bias as:

\begin{align}
\underbrace{\mathbb{E}[Y_{i}(1) \mid D=1] - \mathbb{E}[Y_{i}(0) \mid D=0]}_{\text{Simple Difference in Outcomes (SDO)}} 
&= \underbrace{\mathbb{E}[Y_{i}(1)] - \mathbb{E}[Y_{i}(0)]}_{\text{Average Treatment Effect (ATE)}} \notag \\
&\quad + \underbrace{\mathbb{E}[Y_{i}(0) \mid D=1] - \mathbb{E}[Y_{i}(0) \mid D=0]}_{\text{Selection Bias}} \notag \\
&\quad + \underbrace{(1-\pi)(ATT - ATU)}_{\text{Heterogeneous Treatment Effect Bias}}
\end{align}

where \((1-\pi)\) is the proportion of the population in the control group.[^mixtape413] This decomposition highlights three key components: the true average treatment effect (ATE), selection bias, and heterogeneous treatment effect bias. ATE represents the true causal effect averaged over the entire population. It is the effect we aim to estimate but cannot observe directly due to the fundamental problem of causal inference—we only observe one potential outcome for each unit.The second and third term captures selection and heterogeneous treatment effect bias, which will be discussed below. These biases must be accounted for to estimate the true treatment effect accurately. All the methods in RCM framework we will discuss in the following sections and next chapter aim to address these biases and isolate the causal effect of treatment on the outcome. Let's continue our discussion first discussing assumptions, then these biases and then with the parameters of interest in causal inference.

[^mixtape413]: Detailed derivation of this equation is provided in section 4.1.3 of *Mixtape*. For more details, visit [https://mixtape.scunning.com/04-potential\_outcomes](https://mixtape.scunning.com/04-potential_outcomes).


## Limitations of RCM:

The fundamental problem of causal inference makes ATE, ATT, ATU, LATE, and CATE unknowable since we can observe individuals in only one state (treated or untreated). To estimate these parameters, researchers rely on statistical methods under identifying assumptions by comparing treated and untreated groups. This estimation process raises several critical questions: Were units randomly assigned to treatment? Were the groups sufficiently large and similar? Was the treatment uniformly administered across all individuals? The answers to these questions determine the identification strategy and shape the choice of causal inference methods, which will be covered throughout this book. Each method seeks to overcome the limitations imposed by missing counterfactuals and to estimate treatment effects as accurately as possible given the data and assumptions at hand.

Randomized Controlled Trials (RCTs) are the gold standard for causal inference, ensuring that treatment assignment is independent of potential outcomes. This eliminates selection bias, allowing direct estimation of causal effects with minimal assumptions. However, RCTs are often impractical due to ethical, logistical, or financial constraints. In such cases, researchers turn to quasi-experimental methods like matching, weighting, regression adjustment, instrumental variables, difference-in-differences, and regression discontinuity to approximate randomization and address selection bias.

While the RCM is widely used in modern causal inference, it has limitations. One key criticism, highlighted by Heckman and Vytlacil (2007), its lack of external validity (cannot generalized to new settings or untested programs) and failure to provide insights into underlying mechanisms (treating interventions as "black boxes") limit its generalizability. Angrist and Pischke (2010) argue that causal estimates under RCM are context-specific, requiring synthesis across studies for broader conclusions. Additionally, RCM estimands like ATE, ATT, and ATU may fail to guide marginal policy decisions or account for externalities, such as changes in scale or assignment mechanisms (Heckman et al., 1999).

Alternative frameworks address some of these limitations. The Roy model incorporates self-selection by modeling individual decisions based on expected returns, enabling analysis of treatment effect heterogeneity. Directed Acyclic Graphs (DAGs), introduced by Judea Pearl, use graphical tools and "do-calculus" to represent causal relationships and identify effects. However, both frameworks have trade-offs. Roy models rely on strong parametric assumptions. DAGs are less common in economics due to their complexity, lack of focus on estimation, and challenges in modeling interference and simultaneity (Imbens, 2020).

Finally, while machine learning cannot replace sound experimental design or credible assumptions, it improves causal analysis by improving flexibility, precision, and scalability, particularly in high-dimensional settings. As we transition to the next section on randomization, we will explore how RCTs address selection bias and why they remain the most robust method for identifying causal effects. For settings where randomization is not feasible, subsequent chapters will examine quasi-experimental methods to address the challenges of observational data and provide actionable insights for policy and research.




