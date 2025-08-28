\mainmatter

# Introduction

This chapter introduces the major themes of the book and sets the foundation for how we think about using machine learning (ML) and econometrics to address empirical questions in economics, health, and the social sciences. We begin by distinguishing prediction from estimation, then explore applications of ML across domains, compare the roles of ML and econometrics, and outline core modeling frameworks including statistical versus ML paradigms, parametric and nonparametric methods, and predictive versus causal approaches. We also discuss model selection and introduce simulation as a tool for learning and evaluation. These ideas will be developed and applied throughout the book.

Let’s start with a parable about how perspectives on problems can differ, much like varied views of the same truth.

![](png/aiel.png){width=60% style="display: block; margin-left: auto; margin-right: auto; margin-top: 1em; margin-bottom: 1em;"}

::: {.storybox}   
A group of blind men heard that a strange animal had been brought to their town. None of them had seen or touched one before. Out of curiosity, they said: "We must inspect and understand it by touch, which we are capable of." So, they sought it out, and each of them touched a different part of the animal.

“An animal like a tree!” — said one, holding its leg.

“A wall!” — said another, feeling its broad side. 

“A snake!” — said a third, grasping its trunk.  

“A rope!” — said another, grabbing the tail. 

“A fan!” — said one, feeling the ear.  

“A spear!” — said the last, touching the tusk.

They were all describing the same animal—an elephant—but each had only part of the picture.  
:::

This parable mirrors how we approach machine learning and econometrics: distinct lenses on a shared pursuit of explanation through data. Econometricians debate the core of their field—frequentist or Bayesian, theory-driven or applied, parametric or nonparametric, micro or macro, prediction or causation—yet all wrestle with the same beast: understanding the world through data. ML, too, invites divergent definitions: a computer scientist may envision deep reinforcement learning, an economist sees LASSO for variable selection, while a statistician views it as optimized regression. Each grasps a piece of the truth, their perspectives as varied as touches on an elephant.

Contrary to its name, machine learning is not about autonomous “learning” in any cognitive sense. It is the process of constructing models using data-driven algorithms rooted in statistical principles, with the goal of improving predictive or inferential performance. These models—ranging from linear regressions to decision trees and neural networks—are built through algorithmic estimation and executed on modern computational systems.

Just as a child learns to distinguish cats from dogs by observing many examples, supervised learning uses labeled data to train algorithms that map inputs to outputs. These models power spam filters, voice assistants, and recommendation engines. In this book, however, we emphasize ML’s role in empirical research—particularly in building better estimators, uncovering structure, and strengthening causal analysis.

Within economics, social and health sciences, ML is increasingly used to uncover causal relationships. Whereas prediction focuses on forecasting outcomes, causal inference explores how a variable, intervention, or treatment affects an outcome. Consider questions like: Does a scholarship program boost graduation rates? Does telemedicine reduce hospital readmissions? While ML has transformed domains like image recognition and language processing, our emphasis is on its contribution to applied data analysis—especially in improving estimation, modeling heterogeneity, and enabling flexible, interpretable models. 

We structure this book around the estimation and inference challenges commonly encountered in applied research. A central conceptual anchor is the bias-variance trade-off, which provides a unifying framework for understanding model generalization, overfitting, and regularization.[^otml]

[^otml]: Other theoretical foundations—such as Vapnik-Chervonenkis theory, computational complexity, and Bayesian learning—are important in different contexts (e.g., deep learning, theoretical ML), but less central to the kinds of data and questions we focus on here.

ML methods are generally categorized as:

- Supervised learning – learning from labeled data (e.g., regression, classification)

- Unsupervised learning – discovering patterns without labels (e.g., clustering, dimensionality reduction)  

- Reinforcement learning – adapting through feedback over time 

- Deep learning – modeling with multi-layered neural networks  

Although these paradigms enable a broad range of applications—from robotics to genomics—our focus is on supervised learning, particularly regression and classification, with selected coverage of unsupervised methods such as clustering and dimensionality reduction. Our objective is to demonstrate how ML complements, rather than replaces, econometric approaches. We begin with foundational concepts, proceed through predictive modeling, and culminate with modern approaches to causal inference. The aim is to provide readers with a technically grounded, practically oriented framework for using machine learning in applied research.


## Prediction vs. Estimation

In economics, health, and the social sciences, researchers routinely aim to identify patterns, quantify associations, generate forecasts and predictions, and estimate causal effects. Yet terms such as prediction, extrapolation, forecasting, and estimation are often used interchangeably—particularly in machine learning (ML) contexts. Despite their similarities, these concepts have distinct meanings and serve different analytical purposes.

In the simplest terms, prediction refers to using observed data to make accurate generalizations to unseen or future events. Related concepts include extrapolation, which extends patterns beyond the observed data range, and forecasting, which emphasizes time-indexed predictions based on historical trends—especially common in finance and macroeconomics. In ML, prediction typically involves estimating a function that maps input features (variables or covariates) to outcomes, with performance evaluated on out-of-sample (unseen) data.

On the other hand, social scientists and economists are often concerned with **estimation**, , particularly in a causal framework. Estimation focuses on quantifying the relationship between variables, frequently aiming to recover population-level parameters from sample data, typically emphasizing causal inference. While prediction often focuses on accurately forecasting unit-level outcomes, estimation is concerned with understanding and quantifying relationships, such as determining the causal impact of an intervention or treatment.

Distinguishing between these objectives is critical for selecting appropriate methods and framing research questions. For instance, in clinical settings, accurately predicting patient outcomes may guide decision-making. In contrast, health economists are typically interested in estimating whether a treatment improves outcomes on average across a population.

However, predictive modeling in the social sciences presents unique challenges. Economic agents and institutions adapt in response to predicted outcomes, potentially altering the very patterns being forecast. This critique is formalized in the Lucas Critique, which states:

> "Given that the structure of an econometric model consists of optimal decision rules of economic agents, and that optimal decision rules vary systematically with changes in the structure of series relevant to the decision maker, it follows that any change in policy will systematically alter the structure of econometric models. (Robert Lucas, 1976)"

Unlike natural sciences—where identified relationships often remain stable under consistent conditions—social science contexts frequently involve changing relationships influenced by policy actions or behavioral responses, complicating predictive reliability. Nevertheless, well-calibrated predictions remain valuable. Effective predictions indicate deeper understanding and often require balancing model simplicity and flexibility. 

Finally, real-world scenarios frequently demonstrate the practical importance of individual-level prediction versus average causal estimation. A parent may be more concerned with how an education policy affects their own child than the average student. A physician may prioritize personalized prognostics over population-level inference. Law enforcement agencies often rely on predictive models to allocate resources, reflecting the operational relevance of individual-level forecasting.

In sum, both prediction and estimation play foundational roles in applied empirical work. ML tends to associate prediction with learning a function that performs well on unseen data, while estimation emphasizes parameter identification and causal interpretation. Clearly articulating this distinction helps applied researchers align methodological choices with their analytical objectives.[^PrevsEst] With this conceptual distinction in hand, we now turn to how ML methods are applied in practice—both to improve predictive accuracy and to support causal inference across applied domains.

[^PrevsEst]: For a deeper understanding of the distinctions between prediction and estimation, consider reviewing Bradley Efron's (2020) *Prediction, Estimation, and Attribution*, Galit Shmueli's (2010) *To Explain or To Predict?*, and Tetlock and Gardner's (2015) *Superforecasting: The Art and Science of Prediction*.


## Applications of Machine Learning in Economics and Social Sciences

Machine learning (ML) has increasingly reshaped empirical research in economics and related fields. While commonly associated with prediction, ML’s utility extends well beyond forecasting. When integrated with econometric reasoning, ML improves data utilization, strengthens causal estimation, uncovers treatment heterogeneity, and supports more granular policy evaluation. This section highlights key areas where ML contributes meaningfully to applied research.

A major contribution of ML lies in its ability to extract structured variables from unconventional or unstructured sources. Text, images, and historical documents can now be transformed into quantitative inputs—such as sentiment scores, complexity indices, or classification tags—using tools from natural language processing (NLP) and computer vision. These capabilities have expanded the measurable dimensions of human behavior, institutions, and policy environments.

Although causal inference remains a core strength of traditional econometrics, ML complements these frameworks in powerful ways. Methods such as **double machine learning** and **targeted regularization** facilitate treatment effect estimation in high-dimensional settings. Within established designs—including instrumental variables (IV), difference-in-differences (DiD), regression discontinuity (RD), and synthetic control—ML aids in modeling counterfactuals, selecting control variables, and improving estimation robustness. These hybrid strategies improve both credibility and precision.

Heterogeneity in treatment effects is often central to policy analysis. ML methods such as **causal forests** allow researchers to detect and interpret effect heterogeneity across subgroups—for instance, identifying whether low-income students gain more from education subsidies, or whether specific patient populations benefit differentially from medical interventions. These insights support more targeted and equitable policy design.

In labor and health economics, ML is used to process and analyze large-scale administrative datasets, detect latent patterns, and uncover interaction effects. In industrial organization, ML improves demand estimation by refining discrete choice models, such as logit and nested logit, and improving dynamic discrete choice frameworks. These advances yield better out-of-sample predictions and enable richer counterfactual simulations, supporting firm-level strategy and policy analysis.

In finance and macroeconomics, ML models are applied to tasks such as credit scoring, asset pricing, and economic forecasting. Boosting algorithms, deep learning architectures, and flexible time series methods enable the incorporation of diverse feature sets—from macroeconomic indicators to event-driven variables—to forecast inflation, output growth, and volatility. Central banks and financial institutions increasingly employ ML for risk monitoring, stress testing, and monetary policy analysis.

Social scientists also apply ML to study networks, communication, and behavioral dynamics. Clustering and graph-based algorithms reveal patterns in information diffusion, political polarization, and social influence. Urban and regional economists leverage ML to analyze satellite imagery, transportation flows, and housing data—informing the impact of infrastructure projects or zoning reforms. In environmental economics, ML tools facilitate remote sensing and the detection of deforestation, pollution, and climate vulnerability, aiding in the design of evidence-based environmental policies.

Despite their versatility, ML methods pose challenges. Model interpretability can be limited—especially with black-box algorithms—and results may be sensitive to data quality and algorithmic bias. These concerns are particularly acute in high-stakes or socially sensitive domains. As a result, ML applications in economics and the social sciences require theoretical grounding, domain expertise, and a commitment to transparency and reproducibility.

Ultimately, ML expands the empirical toolkit available to applied researchers. It enables new forms of measurement, improves causal inference, and deepens our understanding of effect heterogeneity. ML does not replace economic theory or classical econometrics; rather, it complements and strengthens them when applied thoughtfully. Throughout this book, we highlight empirical studies that demonstrate these methods in action—including recent applications in health, labor, education, finance, and development studies. Our aim is to illustrate how modern ML methods can be used to answer substantive, policy-relevant questions with rigor and clarity. These real-world applications also raise an important methodological question: when should researchers rely on machine learning, and when is econometric structure essential? That’s the focus of the next section.

## Machine Learning and Econometrics: Complementary Roles

Machine learning (ML) is a powerful tool for analyzing data, but its usefulness depends on the research question. For tasks focused on prediction or classification, ML excels — it handles large datasets, captures complex nonlinear patterns, and adapts well to new data. Supervised learning algorithms like random forests, boosting, and neural networks are especially effective in forecasting outcomes such as disease progression, credit default, or stock prices. Unsupervised methods, including clustering and dimensionality reduction, can reveal hidden structure or groupings in data, offering exploratory insights.

However, when the objective is to understand causal relationships, explain mechanisms, or evaluate policy interventions, traditional econometric methods — such as regression, instrumental variables (IV), difference-in-differences (DiD), and regression discontinuity designs (RDD) — are often more appropriate. These approaches are built around specific assumptions about counterfactuals, selection, and identification that ML algorithms do not address on their own. For example, estimating what would have happened to a treated group in the absence of treatment requires assumptions like unconfoundedness or valid instruments, which cannot be learned purely from the data.

Another challenge is interpretability. Many ML models function as “black boxes,” offering limited visibility into how predictions are generated. This poses difficulties when the goal is not just to predict but to explain — particularly in policy contexts where transparency and accountability matter. Although explainable AI (XAI) techniques such as feature importance scores or SHAP values are improving interpretability, they still lack the clarity and inferential rigor of econometric models designed for hypothesis testing and causal analysis.

ML’s reliance on historical data can also be a limitation. When facing rare or unprecedented events — such as financial crises, pandemics, or natural disasters — models trained on past data may fail. Overfitting is another common pitfall, particularly in flexible models trained on limited samples. This happens when a model captures noise rather than true patterns, performing well on training data but poorly on new observations. Regularization techniques help by penalizing model complexity, while cross-validation tests how well a model generalizes across different data subsets. These tools reflect a broader bias-variance tradeoff: simpler models may underfit, missing important structure (high bias), while overly complex ones may overfit (high variance).

In practice, the choice between ML and econometrics is not either-or. They are most powerful when used together. For example, ML can support causal inference by selecting control variables in high-dimensional settings (e.g., using Lasso before DiD), or by modeling treatment effect heterogeneity with causal forests. Predictive models aid in feature engineering or understanding variable importance, while econometric tools provide the structure needed for credible causal conclusions.

As you move through this book, consider how prediction and causality serve different goals — and how aligning methods with your research question will lead to more credible and useful results. Later chapters will show how recent advances like Double Machine Learning and Generalized Random Forests integrate ML with causal inference frameworks. To understand how these approaches are implemented in practice, we now turn to the foundational modeling frameworks that shape empirical strategies.


## Core Modeling Frameworks

In contemporary data analysis, modeling is not merely a technical task but a strategic decision that shapes how we extract insight from data. This chapter introduces key conceptual frameworks that guide empirical modeling across disciplines. We begin by contrasting statistical modeling and machine learning approaches, outlining their respective goals, assumptions, and practical implications. We then examine the distinction between parametric and nonparametric models, followed by a discussion of predictive versus causal modeling. Finally, we consider essential aspects of model selection, including the balance between complexity, interpretability, and generalization.  Throughout the book, we use simulations to demonstrate concepts and evaluate model performance under known conditions. This strategy not only builds intuition but also makes the theoretical and practical implications of modeling choices more transparent.

### Statistical vs. Machine Learning Paradigms

Modern empirical work often draws on two major modeling paradigms: statistical modeling and machine learning. While they are sometimes treated as interchangeable—especially under the umbrella of “statistical learning”—these approaches differ meaningfully in orientation, methodology, and objective. Understanding these differences is essential for aligning modeling choices with research goals—whether explanatory, predictive, or both.

Statistical modeling is primarily concerned with inference. These models are built on assumptions about the underlying data-generating process and are often motivated by theory or domain knowledge. They aim to identify structural relationships and provide interpretable estimates of parameters—such as treatment effects or associations between covariates and outcomes. These models allow researchers to estimate effect sizes, test hypotheses, and draw generalizable conclusions—particularly in fields such as economics, health, and the social sciences, where inference, transparency, and theoretical coherence are highly valued.

Machine learning, by contrast, is predominantly geared toward prediction. It uses flexible, data-driven algorithms that learn patterns from input-output pairs, often with minimal assumptions about functional form or distributional structure. The primary goal is to estimate a function \( \hat{f}(x) \) that accurately maps inputs to outputs, optimizing predictive accuracy on unseen data. Evaluation emphasizes performance metrics such as out-of-sample loss, cross-validation scores, and classification accuracy. These models excel in high-dimensional or nonlinear settings where traditional models struggle or fail to generalize. 

Although both approaches can be applied to prediction and inference, they differ in prioritization. Statistical models emphasize interpretability and causal structure, while ML models prioritize flexibility and predictive performance—often at the expense of transparency. This leads to trade-offs in applied work, particularly when researchers must balance clarity of interpretation with predictive accuracy.

In many health and social science applications, the dominant tradition remains inferential statistical modeling. Researchers tend not to partition data into training and test sets but rather aim to specify models that reflect theoretical relationships. These models are used for hypothesis testing and estimation, often with a focus on identifying causal effects or quantifying uncertainty around parameter estimates. Typically, they are constructed with substantial input from domain knowledge and involve clear articulation of identification strategies—such as randomized assignment, instrumental variables, or fixed effects—to support valid inference.

In contrast, ML approaches are generally agnostic view of model structure. Their objective is not to model the data-generating process per se, but to capture empirical regularities that support accurate prediction or classification. As data sets grow in size and complexity, ML methods have proven especially valuable for uncovering structure in noisy environments and automating complex tasks.

These contrasting philosophies reflect not only methodological preferences but also disciplinary histories. Statistical modeling evolved from classical statistics and econometrics, emphasizing inference, hypothesis testing, and model transparency. Machine learning emerged from computer science and engineering, with a focus on optimization, scalability, and algorithmic performance.

Leo Breiman’s (2001) influential essay, Statistical Modeling: The Two Cultures, formalized this division. In the “data modeling” culture—characteristic of traditional statistics—analysts begin with a stochastic model, estimate its parameters, and test hypotheses. In contrast, the “algorithmic modeling” culture treats the relationship between inputs and outputs as unknown and potentially complex, focusing on algorithms that achieve high predictive performance without assuming a specific model structure. Breiman’s framing has had a lasting impact on how researchers think about complexity, generalization, and the role of theory in empirical work.

Many applied researchers now adopt hybrid strategies.[^Bayes] For example, ML methods such as Lasso or random forests may be used for variable selection or to identify heterogeneity, followed by more structured statistical modeling for inference. This pragmatic blending allows researchers to leverage the strengths of both paradigms—exploratory flexibility and interpretive clarity—without fully committing to either extreme.

[^Bayes]:Bayesian modeling provides yet another framework that unifies aspects of both traditions by formally incorporating uncertainty and prior beliefs into estimation. While powerful, Bayesian methods are beyond the scope of this book.

In practice, choosing between statistical modeling and machine learning—or combining them—depends on the research question, the nature of the data, and the importance of interpretability versus prediction accuracy. In applied settings, this often means using machine learning tools to explore high-dimensional spaces, identify patterns, or generate predictions, while relying on statistical models to interpret relationships, test hypotheses, or estimate causal effects.

Ultimately, the goal is not to declare one approach superior but to understand their respective strengths and limitations. By building a solid understanding of both paradigms and the trade-offs they entail, researchers are better equipped to select the right tools, interpret results with care, and contribute to meaningful, data-informed decision-making.

In sum, the central challenge in empirical modeling is not choosing between prediction and explanation, but aligning analytical strategies with well-defined research objectives. By understanding the assumptions, strengths, and limitations of different modeling frameworks, researchers can make informed methodological choices that improve both the rigor and relevance of their work.

The following sections examine key modeling distinctions—parametric versus nonparametric, predictive versus causal—and return to Breiman’s “two cultures” in Chapter 9, where we revisit their connection to modern empirical practice.

### Parametric and Nonparametric Models:

Parametric and nonparametric models represent two foundational approaches to empirical modeling in statistics and machine learning. Each offers distinct advantages and limitations, and the choice between them depends on the research question, the structure of the data, and the plausibility of underlying assumptions.

Parametric models rely on explicit assumptions about the functional form of relationships among variables and the distribution of the data-generating process. These models are characterized by a finite set of parameters, estimated using methods such as least squares or maximum likelihood. Examples include linear regression, logistic regression, and polynomial regression. When their assumptions hold, parametric models tend to be efficient, interpretable, and computationally straightforward. However, misspecification—such as assuming linearity when the true relationship is nonlinear—can lead to biased estimates, invalid inference, and poor predictive performance.

Nonparametric models, by contrast, impose minimal structure on the relationship between inputs and outcomes. Rather than specifying a functional form in advance, they allow the data to guide the shape of the function. This flexibility makes nonparametric methods valuable when the true relationship is unknown or inherently complex. Examples include k-Nearest Neighbors (k-NN), kernel density estimators, local polynomial regressions, decision trees, and rank-based statistics such as Spearman’s correlation. These models are particularly useful for detecting nonlinearities, capturing interactions, and handling ordinal or categorical data without transformation.

In statistical terms, parametric modeling begins with an assumed form for an unknown function \( f \), such as \( f(x) = \beta_0 + \beta_1 x \). The model is then estimated using data to recover the parameters \( \beta_0, \beta_1 \). This structure promotes interpretability and parsimony and often requires less data for estimation. However, when the functional form is incorrect, the model can yield misleading conclusions.

Nonparametric methods, in contrast, place no constraints on the form of \( f \). They aim to recover it directly from the data, enabling more flexible estimation of relationships. For instance, kernel regression estimates \( m(x) = E[Y \mid X = x] \) without imposing linearity or a parametric distribution. This flexibility comes at a cost: these models typically require more data, are computationally intensive, and can overfit without appropriate tuning.

Tuning parameters—such as the bandwidth in kernel methods, the number of neighbors in k-NN, or the depth of a decision tree—play a central role in balancing bias and variance in nonparametric models. These settings control model complexity and directly affect generalization. Poor tuning can lead either to underfitting (overly smooth models) or overfitting (models too responsive to noise).

Another consideration is dimensionality. Nonparametric methods often struggle in high-dimensional settings due to the curse of dimensionality, where the data become sparse and local estimation becomes unstable. This limits their effectiveness unless combined with dimensionality reduction, regularization, or variable/feature selection strategies. These issues are revisited in detail in later chapters.

Between the two extremes lie semi-parametric models, which integrate structured parametric components with flexible, data-driven elements. A central example is the partially linear model, which we will revisit throughout the causal inference chapters, particularly in the context of double machine learning and heterogeneity analysis. These models allow researchers to impose structure where theoretical relationships are well understood, while leaving other components unspecified and estimated nonparametrically. Semi-parametric methods provide a practical compromise—preserving interpretability where needed, while accommodating complexity where warranted.

These differences echo the broader contrast outlined by Leo Breiman (2001) in Statistical Modeling: The Two Cultures, previously discussed. Parametric methods align with the data modeling tradition, emphasizing structure and interpretability. Nonparametric approaches reflect the algorithmic tradition, prioritizing flexibility and predictive performance. Increasingly, researchers blend both—using flexible models for exploration or variable selection and structured models for inference and explanation.

In Chapter 9, Parametric Estimation – Basics, we return to classical estimation techniques and their relationship to the data modeling tradition. In Chapter 10, Nonparametric Estimation – Basics, we focus on flexible approaches to estimating conditional expectation functions \( E[Y \mid X = x] \), where no structural assumptions are imposed. These chapters build on the framework outlined here and explore how parametric and nonparametric perspectives interact in modern empirical work.

To summarize, parametric and nonparametric models represent complementary strategies for learning from data. Parametric models offer structure, interpretability, and efficiency but depend on strong assumptions. Nonparametric models provide flexibility and minimal prior restrictions but require careful tuning and larger datasets. Understanding the trade-offs between these approaches—and when to combine them—is central to effective, rigorous empirical analysis.

### Predictive vs. Causal Thinking

Predictive and causal models reflect two fundamentally different goals in empirical analysis. While both rely on observed data, they differ in their estimands, assumptions, evaluation criteria, and implications for decision-making. Distinguishing between these objectives is essential for choosing appropriate methods and correctly interpreting results.

**Predictive modeling** aims to estimate the conditional expectation \( E[Y \mid X] \)—the expected outcome given observed covariates. The objective is to construct a function \( \hat{f}(X) \) that minimizes prediction error on new or unseen data. Performance is typically assessed using out-of-sample metrics such as mean squared error, classification accuracy, or AUC. Predictive models are widely applied in domains such as marketing, finance, and healthcare, where the goal is to forecast outcomes, behaviors, or risks.

These models rely on empirical associations and do not attempt to establish causal relationships. While they can uncover valuable patterns, they offer no insight into the mechanisms generating those patterns. Tools commonly used for prediction include time series forecasting, regularized regression, decision trees, neural networks, and ensemble methods. However, these approaches are susceptible to overfitting—modeling noise instead of signal—especially in complex or high-dimensional settings. Techniques such as cross-validation and regularization are used to mitigate this risk and improve generalization.

**Causal modeling**, by contrast, is concerned with estimating the effect of interventions or exposures, often framed using the potential outcomes framework. A common causal estimand is the average treatment effect (ATE), defined as \( E[Y(1) - Y(0)] \), where \( Y(1) \) and \( Y(0) \) represent the potential outcomes under treatment and control, respectively. Causal questions ask: *What is the expected change in outcome if we intervene on variable \( X \)?*

Identifying causal effects requires strong assumptions—such as unconfoundedness, exclusion restrictions, or monotonicity—and research designs that introduce or approximate exogenous variation. Common approaches include randomized controlled trials (RCTs), natural experiments, instrumental variables (IV), difference-in-differences (DiD), regression discontinuity (RD), and matching techniques. These methods prioritize internal validity over predictive performance.

Importantly, a model with excellent predictive accuracy may fail to uncover causal effects, while a well-identified causal model may perform poorly at forecasting individual outcomes. The central distinction lies in the estimand and purpose: prediction seeks accuracy on future data; causality seeks to answer counterfactual questions.

These objectives also shape how data are used. Predictive modeling emphasizes performance on held-out samples, often via training-test splits or cross-validation. Causal inference focuses on isolating variation that mimics random assignment, often requiring the full sample or carefully defined subgroups to estimate credible treatment effects.

Recent advances increasingly bridge the gap between these paradigms. Methods such as causal forests and double machine learning integrate machine learning into causal estimation. These approaches use ML to model nuisance functions—like propensity scores or outcome regressions—while preserving valid inference under appropriate assumptions.

Ultimately, the appropriate modeling strategy depends on the research question. If the aim is to optimize decisions based on forecasts—e.g., predicting hospital readmissions or credit default risk—predictive modeling is appropriate. If the goal is to evaluate the effect of a policy, treatment, or intervention—e.g., estimating the impact of a job training program on earnings—causal inference is required. Being explicit about one's objective is critical to selecting the right tools and avoiding methodological or interpretive errors.

In summary, predictive and causal models serve distinct purposes. Predictive modeling seeks to minimize error in estimating \( E[Y \mid X] \), relying on flexible algorithms and evaluated through out-of-sample performance. Causal modeling aims to identify counterfactual differences—such as \( E[Y(1) - Y(0)] \)—and requires careful attention to design, assumptions, and identification strategy. Understanding these conceptual and practical differences is essential for credible and relevant empirical analysis. All key concepts introduced here are explored in detail in later chapters.

### Model Selection:

Model selection is a critical step in empirical analysis. It directly influences the accuracy, interpretability, and policy relevance of results. The process involves choosing a model class, specifying its structure, selecting estimation techniques, and aligning these decisions with the underlying objective—whether prediction or causal inference. Each choice entails trade-offs between flexibility and interpretability, complexity and generalizability.

The first consideration typically involves the model family. As discussed earlier, parametric models rely on fixed functional forms and a finite set of parameters, while nonparametric models allow the data to determine the model’s shape. For example, in predicting housing prices, a linear model may assume additive effects for square footage and location, whereas a tree-based model segments the feature space adaptively, capturing nonlinearities and interactions. The appropriate choice depends on whether the structural assumptions are defensible and whether interpretability or adaptability is more important.

Next, the researcher must specify the functional form. A linear model implies constant marginal effects—e.g., a fixed income gain per additional year of education. Adding polynomial or interaction terms introduces nonlinearities and captures diminishing or increasing returns. The key challenge is to enrich the model without overfitting. These specifications should be guided by exploratory analysis and theoretical expectations about the data-generating process.

After selecting a model structure, the next step is choosing an estimation method. In parametric settings, ordinary least squares (OLS) and maximum likelihood estimation (MLE) are standard. In high-dimensional contexts, regularization techniques such as Lasso or Ridge regression help control complexity by shrinking or selecting parameters. Nonparametric models typically require tuning hyperparameters—such as bandwidth, number of neighbors, or tree depth—using cross-validation to manage the bias–variance trade-off.

Model selection also depends heavily on the analytical goal. For prediction, the priority is out-of-sample accuracy, and flexible algorithms with high capacity may be preferred. For causal inference, the emphasis shifts to interpretability, alignment with identification assumptions, and consistency with the counterfactual framework. For example, estimating the effect of education on health may call for a theory-driven model with careful covariate selection, while predicting hospital readmissions may benefit from a data-driven ML model that prioritizes accuracy over interpretability.

Ultimately, model selection is not a purely technical process. It requires domain expertise, theoretical insight, and awareness of context. Researchers must balance statistical assumptions, computational cost, and the need for interpretability, all while remaining aligned with their research questions.

A full treatment of model selection—including evaluation criteria such as cross-validation, out-of-sample error, and model diagnostics—will be provided in Chapter 12: Model Selection. There, we outline how to implement model comparison systematically and rigorously across different modeling contexts.

### The Role of Simulation

Simulation is a foundational tool in modern empirical research. It allows researchers to study model behavior, explore uncertainty, and evaluate estimator performance in settings where analytical solutions are infeasible or the data-generating process is too complex to model directly.

By generating synthetic data under known conditions, simulation makes it possible to assess how different models perform relative to a defined ground truth. This is particularly valuable for illustrating key statistical concepts—such as bias, variance, overfitting, and generalization error—and for comparing competing methods in controlled environments. Throughout this book, we use simulation exercises to show how modeling decisions affect predictive accuracy, estimator properties, and robustness to assumptions.

Simulation is especially useful for evaluating the behavior of estimators in finite samples. Monte Carlo methods, for example, repeatedly sample from a known distribution to study properties like bias and mean squared error. These techniques also help visualize the impact of model complexity, hyperparameter choices, and identification failures—issues we revisit in later chapters on regularization, causal inference, and model selection.

Because simulation makes performance observable under known conditions, it supports not only validation and diagnostics but also pedagogical understanding. It helps connect theory to intuition, enabling researchers and students alike to see how statistical tools behave across different data environments.

## Concluding Remarks

This chapter introduced the central themes of the book: the role of machine learning in applied empirical research, the distinctions between prediction and estimation, the integration of ML with econometric approaches, and the modeling frameworks that guide empirical practice. Along the way, we previewed key tools and concepts—such as causal inference, regularization, simulation, heterogeneity, parametric vs. nonparametric models, and model selection—that are explored in depth throughout the book. These ideas will reappear in both theoretical and applied contexts as we build from foundational principles toward practical implementations. In the next chapter, we begin by formalizing the statistical and machine learning frameworks used to structure modeling decisions across empirical settings.










