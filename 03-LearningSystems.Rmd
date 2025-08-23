# Learning Systems

::: {.storybox}
Imagine a young and curious person named Alex who occasionally feels sick after eating but isn’t sure which specific food is to blame. He struggles with recurring allergic reactions and, driven by a need to identify the cause, sets out on a personal investigation. Wanting to understand his vulnerability, Alex begins to look for patterns and clues.

Data Collection: Each time Alex eats, he writes down everything he consumed that day. He also notes how he felt afterward.

Pattern Recognition: After several weeks of careful tracking, a pattern emerges. Every time he eats food containing garlic, he feels sick within a few hours. On days when he avoids garlic, he generally feels fine.

Making Predictions: Based on this pattern, Alex suspects that garlic might be causing his discomfort. To test this idea, he avoids garlic for a few days and monitors his health. Then, keeping all other foods the same, he reintroduces garlic on another day to see if the symptoms return.

Validation: On the garlic-free days, Alex feels completely fine. But after eating garlic again, the familiar symptoms come back. This strengthens his belief that garlic is the trigger.

Updating the Model: Wanting to be thorough, Alex tests other ingredients like onions and shallots on separate days. Since he experiences no negative reactions, he concludes that the issue seems specific to garlic.
:::

In this example, Alex is essentially behaving like a simple machine learning model. He:

– Collects data (his meals and symptoms)  
– Looks for patterns  
– Makes predictions based on those patterns  
– Validate predictions against actual occurrences.
– Adjust the predictive model considering new or contradictory data.

While Alex's learning process strongly suggests that garlic triggers his symptoms, it's important to recognize the limitations of his informal "model". Just as in any learning model, prediction errors could arise from multiple factors. For instance, there might be times when he consumes garlic but doesn't get sick because of variations in the quantity consumed, or the form in which it's ingested (raw versus cooked). There could also be external factors, like the combination of garlic with other foods, that influence his reaction. It's also possible that, on some days, other confounding variables like stress or a different underlying illness might mask or exaggerate his garlic-induced symptoms. Thus, while Alex feels confident in his findings, he understands that real-life scenarios can introduce unpredictability, making it essential to continually refine and reassess his conclusions.

This personal example helps to illustrate how learning from experience—through observing patterns and testing hypotheses—is the core idea behind machine learning. Machines follow a similar process, though typically with much more data and formalized steps.

Machine learning, which comes primarily from computer science, also focuses on interpreting data, similar to what we discussed in statistical learning in the previous chapter. However, its main emphasis often lies in making accurate predictions or decisions, even if it doesn’t always explain the underlying reasons. You can think of it like an engineer who builds a tool that works reliably, without needing to fully understand every detail of its internal mechanics.

So why is machine learning sometimes referred to as "statistical learning"? Many machine learning techniques are rooted in statistics, and as both fields evolved, their methods and goals started to overlap significantly. Today, the boundary between them is often blurred. For most people, statistical learning can be thought of as a close relative of machine learning. Both aim to learn from data, but they may prioritize different things—statistical learning often seeks to explain, while machine learning tends to optimize prediction. At its core, machine learning is about teaching computers to make decisions or predictions from data, rather than giving them step-by-step instructions.

The goal in machine learning is to build a model with specific settings or parameters that can make accurate predictions on new, unseen data. Later in this book, we’ll walk through the three central steps: prediction, training, and hyperparameter tuning.

Everything begins with data—whether it’s images, text, economic indicators, political polls, public health records, or employment figures. This data is organized and used as training input for a machine learning model. Generally, more data leads to better model performance.

Once the data is prepared, the process follows three key phases:

**Training (or Parameter Estimation) Phase:** Here, the model learns from the training data. There are two main approaches: one searches for the best model by optimizing a chosen quality measure (producing a point estimate), and the other is Bayesian inference, which is beyond the scope of this book. Regardless of the method, the aim is the same—use numerical techniques to estimate parameters that best fit the data.

**Hyperparameter Tuning (or Model Selection) Phase:** In this step, we evaluate different models and their hyperparameters—settings that aren’t directly estimated from the data—to choose the one that performs best on validation or test data. The ultimate goal is to select a model that generalizes well to data it hasn’t seen.

**Prediction (or Inference) Phase:** This occurs when a trained model is applied to test data it hasn’t seen before. At this stage, the structure of the model and settings are already fixed. The task is simply to use the model to generate predictions for new inputs.

A quick clarification: parameters are learned during training—these are internal values like weights in a regression or a neural network. Hyperparameters, on the other hand, are set before training begins, such as the learning rate or the maximum depth of a decision tree. Hyperparameter tuning helps find the best combination of these settings for optimal performance.

Once a model is trained, tuned, and validated, it can be used in a wide range of practical applications. In machine learning, "learning systems" are algorithms or models that improve over time as they are exposed to more data. They aren’t explicitly programmed for each task but instead adapt by identifying patterns and making decisions based on input.

These systems serve different purposes, depending on the goals of the user or organization:

- The **Descriptive** role focuses on summarizing and understanding patterns in existing data. This is useful for uncovering historical trends and insights.  
- The **Predictive** role goes further, using current data to forecast future outcomes.  
- The **Prescriptive** role is the most action-oriented. It not only predicts but also recommends decisions or actions based on the analysis.

For example, industrial economists might use machine learning to suggest optimal pricing strategies, allocate resources efficiently, or make informed hiring decisions—drawing from data such as consumer demand, competitor pricing, product performance, labor market trends, and skill profiles. In healthcare, machine learning models can help identify cost-effective treatments, optimize hospital operations, target preventive interventions for at-risk populations, inform drug pricing, or guide health insurance premium setting—using data on patient outcomes, resource availability, public health patterns, drug efficacy, and insurance claims.

Now that we've introduced the key stages—prediction, training, and hyperparameter tuning—and how machine learning models are used across different sectors, the next subsection will walk through, step by step, how to find, train, and evaluate models that generalize well to unseen data.

## Learning Systems

A machine learning system is a dynamic framework often used in practical machine learning projects. These systems typically follow an iterative cycle involving several key phases. It begins with Data Collection, where datasets are created, maintained, and continuously updated. This step is foundational, as the quality and relevance of the data greatly affect outcomes.

Next is the Experimentation phase. Here, the collected data is explored in depth, hypotheses are developed and tested, and suitable models are identified and evaluated. This stage also includes building training and prediction pipelines. Once a robust model is developed, the deployment phase puts it into use as a part of a real product or system. However, deployment is not the final step. The Operations phase monitors and updates the model, ensuring it adapts to changing patterns and continues to perform reliably.

While all of these phases—data collection, experimentation, deployment, and operations—are a part of a full machine learning system, it’s important to note that they often require different skill sets and are typically handled by professionals from different domains. For example, deployment and operations are more commonly led by engineers or data infrastructure teams, particularly in industry settings. As economists, social scientists, and health researchers, our primary involvement tends to center around data preparation, model development, and evaluation. Still, understanding the full pipeline is valuable for collaboration and communication in interdisciplinary environments.

In this section, we introduce how machine learning systems are structured in practice, the typical steps in building predictive models, and the core theoretical ideas behind learning from data. We also raise key questions about what makes a model good for prediction versus explanation.

Machine learning systems use algorithms and models that improve over time as they process more data. They do not require detailed programming for every task but instead learn patterns and make decisions based on input data. Our focus in this book is on machine learning models, which are the central components of these systems. A model combines data with code to produce predictions or decisions. How do we find, train, and ensure that our models perform well on unseen data?

Here is a step-by-step overview:

**Splitting the Data:** We begin by dividing the dataset into two main parts: a **training set** and a **testing set**.

**Use Training Data Wisely:** Typically, about 80% of the data is used for training. Within this training portion, we split it again into two parts: **estimation data** (used to train the models) and **validation data** (used to evaluate model performance).

**Keep Testing Data Untouched:** The testing data is set aside and not used during training or model selection. It will be used only at the final stage to assess how well the chosen model performs on unseen data.

**Choose Possible Models:** Before training, we select a few model candidates.  
– For **parametric models**, we specify a functional form for the model, excluding the parameters to be estimated.  
– For **nonparametric models**, we select which features/variables to include and decide on the values for any tuning parameters.

**Train the Model:** Each candidate model is trained (fitted) using the **estimation data**, which is usually the larger portion of the training data.

**Check and Compare Models:** After training, we evaluate each model using the **validation data**, which the model hasn't seen before. This helps estimate how well each model generalizes.

**Pick the Best Model:** We select the model that performs best on the validation data.

**Final Training:** We retrain the chosen model using all of the training data—estimation and validation combined—to make full use of available information

**Test:** Finally, we use the untouched **testing data** to estimate how well the trained model generalizes to new data.

It is essential to distinguish between the roles of validation and testing data: validation data guides model selection, while testing data estimates final model performance.

This overview provides a plain-language summary of the typical machine learning workflow. In the sections that follow, we’ll explore these steps using standard technical terminology. If any terms or ideas feel unfamiliar, don’t worry—we’ll explain each of them in more depth as the book continues.

1. The learner has a sample of observations.  This is an arbitrary (random) set of objects or instances each of which has a set of features ($\mathbf{X}$ - features vector) and labels/outcomes ($y$).  We call this sequence of pairs as a training set: $S=\left(\left(\mathbf{X}_{1}, y_{1}\right) \ldots\left(\mathbf{X}_{m}, y_{m}\right)\right)$.  
2. We ask the learner to produce a prediction rule (a predictor or a classifier model), $\hat{f}(x)$, so that we can use it to predict the outcome of new domain points (observations/instances).  
3. We assume that the training dataset $S$ is generated by a data-generating model (DGM) or some "correct" labeling function, $f(x)$. The "true" prediction function, $f(x)$, is unknown to the learner. The learner tries to approximate $f(x)$ using the sample $S$.

4. The learner will come up with a prediction rule, $\hat{f}(x)$, by using $S$, which will be different than $f(x)$.  The quality of the prediction rule is measured by a loss function, $L_{(S, f)}(\hat{f})$, which quantifies the difference between the true function $f(x)$ and the learner's prediction function $\hat{f}(x)$. This is known as the generalization error or risk.  
5. The goal of the algorithm is to find $\hat{f}(x)$ that minimizes the error with respect to the unknown $f(x)$. The key point here is that, since the learner does not know $f(x)$, it cannot calculate **the loss function**.  However, it calculates the training error also called as the **empirical error** or the **empirical risk**, which is a function that defines the difference between $\hat{f}(x)$ and $y_i$.  
6. Hence, the learning process can be defined as coming up with a predictor $\hat{f}(x)$ that minimizes the empirical error.  This process is called **Empirical Risk Minimization** (ERM).  
7. Now the question becomes what sort of conditions would lead to bad or good ERM?  

If we use the training data (in-sample data points) to minimize the empirical risk, the process can lead to $L_{(S, f)}(\hat{f}) = 0$.  This problem is called **overfitting** and the only way to rectify it is to restrict the number of features in the learning model.  The common way to do this is to "train" the model over a subsection of the data ("seen" or in-sample data points) and apply ERM by using the test data ("unseen" or out-sample data points). Since this process restricts the learning model by limiting the number of features in it, this procedure is also called **inductive bias** in the process of learning.  

There are always two "universes" in a statistical analysis: the population and the sample.  The population is usually unknown or inaccessible to us. We consider the sample as a random subset of the population.  Whatever the statistical analysis we apply almost always uses that sample dataset, which could be very large or very small.  Although the sample we have is randomly drawn from the population, it may not always be representative of the population.  There is always some risk that the sampled data happens to be very unrepresentative of the population.  Intuitively, the sample is a window through which we have partial information about the population.  We use the sample to **estimate** an unknown parameter of the population, which is the main task of **inferential statistics**.  Or, we use the sample to develop a **prediction rule** to predict unknown population outcomes.  

When the outcome is numeric (non-binary), we often use a loss function such as the mean squared error (MSE) to assess the quality of a predictor or an estimator.[^bin] We use the term $\hat{f}(x)$ to refer to either. But can a good estimator also be a good predictor? In the next chapter, we will see that an estimator with a small estimation error can still perform better in prediction. Why does that happen? We will explore this question in detail in the next chapter. 

[^bin]:For binary outcomes, classification errors—another type of loss—become relevant. We will cover those in Chapter 10.

In the next sections, we will carefully examine the difference between estimation and prediction, and explore under what conditions models generalize well to unseen data.
