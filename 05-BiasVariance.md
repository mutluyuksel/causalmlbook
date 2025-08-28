# Bias-Variance Trade-off

In this chapter, we explore the bias-variance tradeoff and its critical role in model performance. In this chapter, we still assume prediction functions (i.e. we assume we know the prediction functions in simulations) to find MSPE using the given sample dataset. We use single data set to calculate MSPE for that given data set (in-sample MSPE for a test data). In this chapter, we show calculating bias-variance trade-off for each assumed function using the sample data. Bias-Variance Trade-off, while complex, is reflected in everyday life situations, offering insights into its practical significance. We begin with the story of Baran, whose experiences in planning his biweekly work trips vividly illustrate this tradeoff.

::: {.storybox} 
Baran aims to catch a flight for his biweekly work trips, with the gate closing at 8 AM. During his first year, he developed a routine to leave his house around 7:00 AM, planning to depart either 30 minutes before or after this time. However, he discovered that this approach sometimes led to him missing his flight. Even though his average or expected departure time from home was 7:00 AM, this process occasionally resulted in arrival times after the flight's gate closure, making his goal of catching the flight more error-prone. This approach represented a scenario with higher variance and lower bias, leading to costly errors concerning his airport arrival time and missed flights. In contrast, with his revised preparation routine, Baran aimed to leave around 7:00 AM but allowed a margin of 10 minutes either way. Consequently, his average departure time shifted to 7:05 AM, introducing a 5-minute bias. However, this new strategy led to more consistent success in catching his flight, reflecting higher bias but lower variance.  This approach nearly eliminated all instances of missing the flight. This whole experience demonstrates the bias-variance tradeoff. In the first case, a preparation process with higher variance and lower bias resulted in occasional flight misses. In the second case, adopting a process with reduced variance, even though it introduces a degree of bias, ultimately improves the overall result. Baran's experience illustrates that consistently opting for lower bias (even unbiased), is not always the best strategy to predict his arrival time, as a balance between bias and variance can lead to more effective real-world results.
:::

As seen from Baran's narrative, the bias-variance tradeoff is not an abstract statistical concept but a tangible factor in our everyday decision-making processes. Grasping this balance is key to understanding how to craft models that are not only accurate but also reliable across various situations. In the next chapter, we'll explore another critical concept in machine learning: overfitting. This concept is also very crucial for crafting precise and dependable models. We will explore how overfitting relates to and differs from the bias-variance tradeoff. 

We already discuss bias and variance of estimator and predictor, and decomposition of MSE for estimation and MSPE for prediction in the previous chapter. Now, Lets discuss what is bias and variance and trade-off between them in predictive models. The bias-variance tradeoff is a fundamental concept in statistical learning and machine learning, which deals with the problem of minimizing and balancing two sources of error that can affect the performance of a model. Bias occurs from oversimplifying the model, leading to underfitting and missing relevant relations in the data. Variance occurs from overcomplicating the model, leading to overfitting and capturing random noise instead of the intended outputs. High bias results in poor performance on training data, while high variance causes poor generalization to new data. The goal is to find a balance, creating a model that is complex enough to capture true patterns but simple enough to generalize well. This balance is achieved through choosing appropriate model complexity, using techniques like cross-validation, applying regularization, and refining data or features.

Following is a formal discussion of bias-variance tradeoff. 

## Bias-Variance Trade-off 

In this section we will discuss the formal definition of bias-variance tradeoff. We will also discuss the Mean Squared Prediction Error (MSPE) and its components, including bias, variance, reducible error, and irreducible error. We will explore how these components interact to influence the overall prediction accuracy of a model.

To remind,our main task is prediction of an outcome, Y using the data (more accurately test/train data). To "predict" Y using features X, means to find some $f$ which is close to $Y$. We assume that $Y$ is some function of $X$ plus some random noise.

\begin{equation}
  Y=f(X)+ \epsilon
\end{equation}
where $\mathbb{E}[\epsilon] = 0$ and $\mathbb{V}[\epsilon] = \sigma ^ 2$. In this formulation, we call $f(X)$ the signal and $\epsilon$ the noise, i.e irreducible error. Moreover, we can never know real $f(X)$. Thus our goal becomes to finding some \(\hat{f(X)}\) that is a good estimate of the regression function $f(X)$. There will be always difference between real $f(X)$ and $\hat{f(X)}$. That difference is called reducible error. To be able to decrease reducibl error as well as to obtain a better prediction function, $\hat{f(X)}$, which is an estimate of unknown, $f(X)$, we can use multiple sample data, which is random from population data, and estimate a function and find its expected value. In chapter 5, we showed the MSPE and their decomposition. Using various samples we find different estimate of function , we attempt to decrease expected MSPE.  We want to emphasize eventhough process looks like similar estimate unknown parameters using different representative samples to estimate parameter with each sample and then find the expected value of parameters. The main difference is instead of assuming same function to estimate parameters like in OLS, Here we estimate different prediction function with each sample and try to estimate unknown function by calculating expected value of the estimated functions. , We find a good estimate of $f(X)$ by reducing the expected mean square of error with each test data as much as possible ( when Y is continuous. when it is binaty, we will use another method which we will cover in classification chapters in detail). To be able to have a variance of prediction functions, we need to use different samples to estimate the prediction function thus have different prediction functions. Below, we will show step by step the bias-variance tradeoff for different polynomial degrees using the same sample data.

Thus, we can write MSPE for prediction function using one training sample (for test data, validation sample):
  
\begin{equation}
  \operatorname{MSPE}(Y-\hat{f(X)})^{2})= (\text{Bias}[\hat{f(X)}])^{2}+\text{Var}[\hat{f(X)}]+\sigma^{2}
\end{equation}
  
Then, Mean Square of Prediction Error (using multiple test data) can be written as:
  
\begin{align}
\mathbb{E}[\text{MSPE}] 
&= \mathbb{E}\left[(Y - \hat{f}(X))^2\right] \notag \\
&= \underbrace{\mathbb{E}\left[(f(x) - \mathbb{E}[\hat{f}(x)])^2\right]}_{\operatorname{bias}^2(\hat{f}(x))} 
+ \underbrace{\mathbb{E}\left[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2\right]}_{\operatorname{var}(\hat{f}(x))} 
+ \sigma^2
\end{align}
where $\sigma^{2}=E[\varepsilon^{2}]$
      
More formally, we can write this as 
  
\begin{align}
      \text{MSPE}\left(Y, \hat{f}(X)\right) = 
        \mathbb{E}_{X, Y, \mathcal{D}} \left[  (Y - \hat{f}(X))^2 \right] =  \notag \\
          \underbrace{\mathbb{E}_{X} \left[\text{bias}^2\left(\hat{f}(X)\right)\right] + \mathbb{E}_{X} \left[\text{var}\left(\hat{f}(X)\right)\right]}_\textrm{reducible error} + \sigma^2
\end{align}
          
  where Y is the outcome data we observe from the seen data and want to predict the ones from unseen data, $\hat{f}(X)$ is the prediction function, $\mathcal{D}$ is random sample from $(X,Y)$, $\sigma^2$ is the irreducible error, $\text{bias}^2\left(\hat{f}(X)\right)$ is the squared bias of the prediction function, $\text{var}\left(\hat{f}(X)\right)$ is the variance of the prediction function, and $\text{reducible error}$ is the sum of the squared bias and variance of the prediction function. The reducible error represents the error that can be minimized through model selection, while the irreducible error is the error that cannot be reduced due to the inherent noise in the data. The goal is to find a model that minimizes the reducible error while balancing the bias and variance of the prediction function. To simplify we donot use subscripts for the expected value of the functions.
        
  The expected mean-squared prediction error (MSPE) on the validation/training sample consists of the sum of the squared bias of the fit and the variance of both the fit and the error/noise term. The error term $\sigma^2$, also referred to as irreducible error or uncertainty, represents the variance of the target variable $Y$ around its true mean $f(x)$. It is inherent in the problem and does not depend on the model or training data. When the data generation process is known, $\sigma^2$ can be determined. Alternatively, estimation of $\sigma^2$ is possible using the sample variance of $y$ at duplicated (or nearby) inputs $x$.
        
   **Variance** is the amount by which $\hat{f(X)}$ could change if we estimated it using different test/training data set. $\hat{f(X)}$ depends on the training data. (More complete notation would be $\hat{f}(X; Y_{train},X_{train})$ )If the $\hat{f(X)}$ is less complex/less flexible, then it is more likely to change if we use different samples to estimate it. However, if $\hat{f(X)}$ is more complex/more flexible function, then it is more likely to change between different test/training samples. 
        
  For instance, Lets assume we have a data set (test or training data) and we want to "predict" Y using features X, thus estimate function $f(X)$. and lets assume we have 1-degree and 10-degree polynomial functions as a potential prediction functions. We say $\hat{f(X)}=\hat{\beta_{0}} + \hat{\beta_{1}} X$ is less complex than 10-degree polynomial $\hat{f(X)}=\hat{\beta_{0}} + \hat{\beta_{1}} X...+ \hat{\beta_{10}} X^{10}$ function. As, 10-degree polynomial function has more parameters $\beta_{0},..\beta_{10}$, it is more flexible. That also means it has high variance (As it has more parameters, all these parameters are more inclined to have different values in different training data sets). Thus, a prediction function has high variance if it can change substantially when we use different training samples to estimate $f(X)$. We can also say less flexible functions (functions with less parameters) have low variance. Low variance functions are less likely to change when we use different training sample or adding new data to the test sample. We will show all these with simulation in overfitting chapter as well.     
        
  **Bias** is the difference between the real! prediction function and expected estimated function. If the $\hat{f(X)}$ is less flexible, then it is more likely to have higher bias. We can think this as real function(reality) is always more complex than the function approximates the reality. So, it is more prone to have higher error, i.e. more bias.  
        
   In the context of regression, Parametric models are biased when the form of the model does not incorporate all the necessary variables, or the form of the relationship is too simple. For example, a parametric model assumes a linear relationship, but the true relationship is quadratic. In non-parametric models when the model provides too much smoothing.
        
   There is a bias-variance tradeoff. That is, often, the more bias in our estimation, the lesser the variance. Similarly, less variance is often accompanied by more bias. Flexible(i.e. complex) models tend to be unbiased, but highly variable. Simple models are often extremely biased, but have low variance.
        
  So for us, to select a model that appropriately balances the tradeoff between bias and variance, and thus minimizes the reducible error, we need to select a model of the appropriate flexibility for the data. However, this selected model should not overfit the data as well which we will discuss in the next chapter.\footnote{ Read [https://threadreaderapp.com/thread/1584515105374339073.html] and [https://www.simplilearn.com/tutorials/machine-learning-tutorial/bias-and-variance]}
        
        
### Simulation for Polynomial Regression
        
   Remember that in real cases, we have data set which consist from outcome Y and features X. We can use different samples to estimate the prediction function, $\hat{f(X)}$, and then find the expected value of the estimated functions. We can obtain these samples by splitting data using different methods which we will discuss in future.   In this simulation, we will use the same sample data to estimate the prediction function for different polynomial degrees. We will use polynomial regression to estimate the function with different polynomial degrees ranging from 1 to 10. We will calculate the bias, variance, reducible error, and irreducible error for each polynomial degree. We will also calculate the expected in-sample MSPE for each polynomial degree. The simulation will provide insights into how the bias-variance tradeoff influences the prediction accuracy of the model and the importance of selecting an appropriate model complexity. We want to remind the prediction function is polynomial function and we know the real function in this simulation. In real cases, we do not know the real prediction function.
        
  First lets discuss data generation process. We start by defining a fixed set of input values, \( X \), drawn from a normal distribution with mean 0 and standard deviation 1. These values are consistent across repeated samples to ensure comparability. Using these input values, we simulate a true function, \( f \), which represents the underlying relationship between \( X \) and the dependent variable \( Y \). The true function is specified as a cubic polynomial with coefficients \( b_0 = 1 \), \( b_1 = 2 \), \( b_2 = -2 \), and \( b_3 = 3 \). To remind, this true function is unknown in reality (outside simulation) and this is the prediction function we are trying to estimate. Additionally, random errors drawn from a normal distribution with mean 0 and standard deviation \( \sigma \), which is 8 in this simulation, are added to the true function to generate the observed values of \( Y \).
        
  If we want to replicate the simulation using the new data and function that we described in section 5.4, which demonstrates reducible and irreducible errors as well as MSPE, the results below will be observed. The variance is zero because we use the same sample (both X and Y) and only one function for prediction; therefore, there is no variance in the prediction function. The irreducible error is close to 60, but it can approach 64 if you increase \(𝑛\) in the simulation program. These results can be achieved by changing the number of simulations, \(M\), to 1 in the simulation program provided below. We have presented these results here because some sources use this simulation to illustrate the bias-variance tradeoff. However, to estimate the prediction function more accurately using multiple data sets, we will need more than one function, which we discuss below. 
        

``` r
        # Function for X - fixed at repeated samples
        xfunc1 <- function(n) {
          set.seed(123)
          x_1 <- rnorm(n, 0, 1)
          return(x_1)
        }
        # Defining a function for simulation 
        simmse <- function(M, n, sigma, poldeg) {
          x_1 <- xfunc1(n) # Repeated X's in each sample
          # Containers
          MSPE <- rep(0, M)
          Irrerr <- rep(0, M)
          yhat <- matrix(0, M, n)
          olscoef <- matrix(0, M, poldeg + 1)
          ymat <- matrix(0, M, n)
          xmat <- matrix(0, M, n) # to store x values
          f_values <- matrix(0, M, n) # to store f(x) values
          # Loop for samples
          for (i in 1:M) {
            b0 <- 1; b1 <- 2; b2 <- -2; b3 <- 3
            f <- b0 + b1 * x_1 + b2 * (x_1^2) + b3 * (x_1^3)
            RandomError <- rnorm(n, 0, sigma)
            y <- f + RandomError
            Irrerr[i] <- mean((y - f)^2)
            samp <- data.frame("y" = y, x_1)
            xmat[i, ] <- x_1
            f_values[i, ] <- f
            # Estimator
            ols <- lm(y ~ poly(x_1, degree = poldeg, raw = TRUE), samp)
            olscoef[i, ] <- ols$coefficients
            # Yhat's: This is equivalent to fhat for OLS
            yhat[i, ] <- predict(ols, newdata = samp)
            # MSPE - That is, residual sum of squares
            MSPE[i] <- mean((ols$residuals)^2)
            ymat[i, ] <- y
          }
          list(MSPE = MSPE, yhat = yhat, sigma = sigma, 
               olscoef = olscoef, f = f, ymat = ymat, xmat = xmat,
               f_values = f_values, Irrerr = Irrerr)
        }
        # Running different fhat with different polynomial degrees
        # Choose certain parameters: M, n, Maxpoldeg, sigma
        M <- 1
        n <- 100
        Maxpoldeg <- 10
        sigma <- 8
        results <- lapply(1:Maxpoldeg, function(degree) 
          simmse(M, n, sigma, degree))
        # Table
        tab <- matrix(0, Maxpoldeg, 5)
        row.names(tab) <- paste0("PolyDeg", 1:Maxpoldeg)
        colnames(tab) <- c("Bias^2", "Variance", "Red.Err.", 
                           "Irreduc.Err.", "In-sample MSPEs")
        f <- results[[1]]$f
        fmat <- matrix(f, nrow = M, ncol = n, byrow = TRUE)
        # Bias^2 = (mean(yhat) - f)^2 = mean((mean(fhat) - f)^2)
        tab[, 1] <- sapply(results, function(res) 
          mean((apply(res$yhat, 2, mean) - f)^2))
        # Var(yhat) = var(fhat) = mean((fhat - mean(fhat))^2)
        tab[, 2] <- sapply(results, function(res) 
          mean(apply(res$yhat, 2, function(x) 
            mean((x - mean(x))^2))))
        # Reducible Error = mean((f - fhat)^2)
        tab[, 3] <- sapply(results, function(res) 
          mean(colMeans((fmat - res$yhat)^2)))
        # Irreducible error = mean((y - f)^2)
        tab[, 4] <- sapply(results, function(res) mean(res$Irrerr))
        # MSPE = mean((y - yhat)^2)
        tab[, 5] <- sapply(results, function(res) 
          mean(colMeans((res$ymat - res$yhat)^2)))
        knitr::kable(round(tab, 4), align = "c", 
                     caption = "In-Sample MSPE decomposition of a function")
```



Table: (\#tab:bias1)In-Sample MSPE decomposition of a function

|          | Bias^2  | Variance | Red.Err. | Irreduc.Err. | In-sample MSPEs |
|:---------|:-------:|:--------:|:--------:|:------------:|:---------------:|
|PolyDeg1  | 23.3361 |    0     | 23.3361  |   59.9858    |     86.5760     |
|PolyDeg2  | 20.5471 |    0     | 20.5471  |   59.9858    |     79.5635     |
|PolyDeg3  | 1.5780  |    0     |  1.5780  |   59.9858    |     58.4078     |
|PolyDeg4  | 2.2781  |    0     |  2.2781  |   59.9858    |     57.7078     |
|PolyDeg5  | 2.2840  |    0     |  2.2840  |   59.9858    |     57.7018     |
|PolyDeg6  | 2.3913  |    0     |  2.3913  |   59.9858    |     57.5945     |
|PolyDeg7  | 2.7805  |    0     |  2.7805  |   59.9858    |     57.2053     |
|PolyDeg8  | 4.2395  |    0     |  4.2395  |   59.9858    |     55.7463     |
|PolyDeg9  | 4.2399  |    0     |  4.2399  |   59.9858    |     55.7459     |
|PolyDeg10 | 4.7932  |    0     |  4.7932  |   59.9858    |     55.1926     |
        
To be able to estimate this unknown true function, we fit polynomial regression models of varying degrees (ranging from 1 to 10) to the simulated data. The polynomial regression models are estimated using ordinary least squares (OLS) regression, with the degree of the polynomial specified as a parameter. The fitted models, denoted as \( \hat{f} \), are used to predict the values of \( Y \) based on the input \( X \). We calculate the mean squared prediction error (MSPE) for each model, which measures the average squared difference between the predicted values and the actual observed values. This provides an indication of the prediction accuracy of each model.

Using this setup, we generate 50 datasets, $\mathcal{D}$, with a sample size $n = 100$ and fit ten models.

\[
\hat{f}_1(x) = \hat{\beta}_0 + \hat{\beta}_1 x
\]

\[
\hat{f}_2(x) = \hat{\beta}_0 + \hat{\beta}_1 x + \hat{\beta}_2 x^2
\]

\[
\hat{f}_3(x) = \hat{\beta}_0 + \hat{\beta}_1 x + \hat{\beta}_2 x^2 + \hat{\beta}_3 x^3
\]

\[
\vdots
\]

\[
\hat{f}_{10}(x) = \hat{\beta}_0 + \hat{\beta}_1 x + \hat{\beta}_2 x^2 + \ldots + \hat{\beta}_{10} x^{10}
\]

So, using dataset1, we estimate beta parameters for each polynomial degree. Then we use these beta parameters to estimate the prediction function for each polynomial degree. We repeat this process for 50 datasets. We calculate the expected value of the estimated prediction functions. We calculate the bias, variance, reducible error, and irreducible error for each polynomial degree. We also calculate the expected in-sample MSPE for each polynomial degree. We will explain this process step by step below.



``` r
# Function for X - fixed at repeated samples
xfunc2 <- function(n) {
  set.seed(123)
  x_1 <- rnorm(n, 0, 1)
  return(x_1)
}
# Defining a function for simulation 
simmse <- function(M, n, sigma, poldeg) {
  x_1 <- xfunc2(n) # Repeated X's in each sample
  # Containers
  MSPE <- rep(0, M)
  Irrerr <- rep(0, M)
  yhat <- matrix(0, M, n)
  olscoef <- matrix(0, M, poldeg + 1)
  ymat <- matrix(0, M, n)
  xmat <- matrix(0, M, n) # to store x values
  f_values <- matrix(0, M, n) # to store f(x) values
  # Loop for samples
  for (i in 1:M) {
    b0 <- 1; b1 <- 2; b2 <- -2; b3 <- 3
    f <- b0 + b1 * x_1 + b2 * (x_1^2) + b3 * (x_1^3)
    RandomError <- rnorm(n, 0, sigma)
    y <- f + RandomError
    Irrerr[i] <- mean((y - f)^2)
    samp <- data.frame("y" = y, x_1)
    xmat[i, ] <- x_1
    f_values[i, ] <- f
    # Estimator
    ols <- lm(y ~ poly(x_1, degree = poldeg, raw = TRUE), samp)
    olscoef[i, ] <- ols$coefficients
    # Yhat's: This is equivalent to fhat for OLS
    yhat[i, ] <- predict(ols, newdata = samp)
    # MSPE - That is, residual sum of squares
    MSPE[i] <- mean((ols$residuals)^2)
    ymat[i, ] <- y
  }
  list(MSPE = MSPE, yhat = yhat, sigma = sigma, olscoef = olscoef, f = f,
       ymat = ymat, xmat = xmat, f_values = f_values, Irrerr = Irrerr)
}
# Running different fhat with different polynomial degrees
# Choose certain parameters: M, n, Maxpoldeg, sigma
M <- 50
n <- 100
Maxpoldeg <- 10
sigma <- 8
results <- lapply(1:Maxpoldeg, function(degree) simmse(M, n, sigma, degree))
# Table
tab <- matrix(0, Maxpoldeg, 5)
row.names(tab) <- paste0("PolyDeg", 1:Maxpoldeg)
colnames(tab) <- c("Bias^2", "Variance", "Red.Err.", 
                   "Irreduc.Err.", "In-sample MSPEs")
f <- results[[1]]$f
fmat <- matrix(f, nrow = M, ncol = n, byrow = TRUE)
# Bias^2 = (mean(yhat) - f)^2 = mean((mean(fhat) - f)^2)
tab[, 1] <- sapply(results, function(res) 
  mean((apply(res$yhat, 2, mean) - f)^2))
# Var(yhat) = var(fhat) = mean((fhat - mean(fhat))^2)
tab[, 2] <- sapply(results, function(res) 
  mean(apply(res$yhat, 2, function(x) mean((x - mean(x))^2))))
# Reducible Error = mean((f - fhat)^2)
tab[, 3] <- sapply(results, function(res) 
  mean(colMeans((fmat - res$yhat)^2)))
# Irreducible error = mean((y - f)^2)
tab[, 4] <- sapply(results, function(res) mean(res$Irrerr))
# MSPE = mean((y - yhat)^2)
tab[, 5] <- sapply(results, function(res) 
  mean(colMeans((res$ymat - res$yhat)^2)))
knitr::kable(round(tab, 4), align = "c", caption = "InSample MSPE decomposition")
```



Table: (\#tab:bias2)InSample MSPE decomposition

|          | Bias^2  | Variance | Red.Err. | Irreduc.Err. | In-sample MSPEs |
|:---------|:-------:|:--------:|:--------:|:------------:|:---------------:|
|PolyDeg1  | 22.4610 |  1.1039  | 23.5650  |   63.2363    |     87.7267     |
|PolyDeg2  | 19.0787 |  1.6748  | 20.7535  |   63.2363    |     82.9245     |
|PolyDeg3  | 0.1283  |  2.1724  |  2.3007  |   63.2363    |     60.9356     |
|PolyDeg4  | 0.1292  |  2.9782  |  3.1075  |   63.2363    |     60.1289     |
|PolyDeg5  | 0.1584  |  3.5336  |  3.6919  |   63.2363    |     59.5444     |
|PolyDeg6  | 0.1692  |  4.0437  |  4.2129  |   63.2363    |     59.0234     |
|PolyDeg7  | 0.1695  |  4.6262  |  4.7957  |   63.2363    |     58.4406     |
|PolyDeg8  | 0.1730  |  5.3350  |  5.5080  |   63.2363    |     57.7283     |
|PolyDeg9  | 0.2827  |  6.0106  |  6.2933  |   63.2363    |     56.9430     |
|PolyDeg10 | 0.2907  |  6.7212  |  7.0118  |   63.2363    |     56.2245     |


We explore the simulation designed to analyze the MSPE and its components.  Initially, we generate a fixed set of predictor values, \(x\), from a normal distribution with mean zero and standard deviation one, using a seed to ensure reproducibility. The primary simulation function takes parameters: number of simulations \(M\), sample size \(n\), standard deviation of the error term, \(\sigma\), and polynomial degree \(poldeg\). This function maintains consistency by using the same \(x\) values across simulations. Within this function, we initialize containers for storing results like MSPE, irreducible error, predicted values, OLS coefficients, and actual values of the dependent variable and predictor. 

For each simulation, we define a true polynomial relationship between \(y\) and \(x\), add random error to create observed \(y\) values, and calculate the irreducible error as the mean squared difference between observed and true \(y\) values. We fit an OLS regression to the data, store the coefficients, calculate the predicted values, and compute bias squared, variance, reducible and irreducible error to decompose the MSPE. We also compute the MSPE.

By analyzing simulation results, we find that higher-degree polynomial models tend to have lower MSPE due to their ability to better fit the data. However, this comes at the cost of increased model complexity and potential overfitting. Polynomial Degree 3 exhibits the smallest reducible error, at 2.3007, suggesting it as the "true" prediction function. However, it is important to note that we cannot solely rely on reducible error for model validation, as it requires the true function, which is unobservable in practice. We can only calculate the MSPE with real data, and finding the lowest MSPE does not conclusively identify the true model because we must also consider the risk of overfitting, which will be discussed in the next chapter. As seen in this simulation, Polynomial Degree 10 has the lowest MSPE.

To visualize the results and illustrate each step in the simulation process, including how we calculate the MSPE and its components, we present figures from the initial three datasets (each containing 100 data points) for polynomial degrees 1 and 10. It is evident that as the degree of the polynomial prediction function increases, so does the variance of the function. This is noticeable in the increased "wiggly" appearance of the 10th-degree prediction function. As the complexity of the function increases—reflected by a higher number of parameters—the variance of the prediction function also rises. <!-- In the figures, light blue represents polynomial degree 1, green represents degree 3, and red represents degree 10.-->In the figures, gray lines represent polynomial degree 1 and 3, while darker gray corresponds to degree 10.\footnote{Polynomial degree is set in stat\_smooth(...). The plots are generated using create\_plot(results, degree, color), where the degree is 1, 3, or 10.}







Additionally, we generate plots that display the fitted polynomial regression lines for degrees 1,3 and 10 separately across all 50 different datasets (training data sets). We plot the 50 fitted regression lines for polynomial degree 1 \(\hat{f}\) in a single graph, along with the expected \(\hat{f}\), which is the average of the 50 fitted regression lines. We do the same for ploynomial degree 3 and 10. You can visualize different polynomial functions by chooosing M- number of data set, n-number of observations in each data set, standard deviation, or polynomial degree in command, results_deg1 <- simmse(50, 100, 8, 1), or changing betas or function f. From these figures, we can observe that when degree of polynomial is 1, variation in all prediction functions is lower than when polynomial degree is 10. As seen when polynomial degree is 10, different prediction functions obtained by different training sets have more variation, more "wiggly" as complexity of function increases, i.e. has more parameters which may lead vary. The bold black lines in each figure indicates expected value these 50 different fhats which we will show how  we calculate step  by step below. Most sources do not provide this detailed explanation. This helps illustrate the trade-off between model complexity and prediction accuracy, emphasizing the importance of selecting an appropriate model that balances bias and variance.





To demonstrate how the simulation program operates, we will now explain the program step-by-step for a polynomial function of degree 3. Keep in mind that the simulation follows the same steps for polynomial degrees ranging from 1 to 10 (the maximum polynomial degree can be adjusted).

First, run the `simmse` function, starting from the `xfunc2` line at the top, up to selecting parameters in the snippet from the program above. Then, execute the simulation for a polynomial degree of 3 and calculate all steps by running the program snippets provided below.





To run the simulation, we set the following parameters: \( M = 50 \), \( n = 100 \), \( \text{poldeg} = 3 \), and \( \sigma = 8 \). These parameters define the number of simulations, the sample size, the degree of the polynomial used in the regression model, and the standard deviation of the error term, respectively. Additionally, we have a flag `show_all` to control the display of results, which is set to `FALSE` by default.

After defining these parameters, we call the `simmse` function to perform the simulation. This function returns several results, including the predictor values (\(x\)), the true function values (\(f(x)\)), and the observed values (\(y\)). As we mentioned, we generated a fixed set of 100 input values, \(X\), drawn from a normal distribution with mean 0 and standard deviation 1. These values of \(X\) are consistent across repeated samples to ensure comparability in this simulation. However, we can use a different set of \((X_{i}, Y_{i})\) as training samples. Using these input values, we generated the true function, \(f\), which represents the underlying relationship between \(X\) and the dependent variable \(Y\). The true function is specified as \(f_3(x) = 1 + 2x - 2x^2 + 3x^3\). To remind, this true function is unknown in reality (outside simulation) and this is the prediction function we are trying to estimate, and these values can be changed within the `simmse` function. Additionally, random errors drawn from a normal distribution with mean 0 and standard deviation \(\sigma\), which is 8 in this simulation (which makes the irreducible error its square 64), are added to the true function to generate the observed values of \(Y_{\text{sim1}}\).

For instance, \(x_{1} = -0.5604756\) then \(f(x_{1}) = -1.2774088\) using the specified true function. Then we add the 1st value from the 100 random errors generated for simulation 1 (these error terms are not reported in this simulation) and obtain \(y(x_{1}) = f(x_{1}) + \epsilon_{1_{\text{sim1}}} = -6.9606613\). We do this for all 100 values of \(X\) in simulation 1. Repeating this process 50 times for each simulated error term, we generated 50 different training datasets such as \((X,Y_{\text{sim1}})\),..., \((X,Y_{\text{sim50}})\). We then extract and round the \(x\), \(f(x)\), and \(y\) values from the simulation results for better readability. These values are organized into matrices for further analysis. Finally, we combine the \(x\), \(f(x)\), and \(y\) matrices into a single output matrix and print it below.



``` r
# Running the simulation
M <- 50
n <- 100
poldeg <- 3
sigma <- 8
show_all <- FALSE  # Set to TRUE to display all results

results <- simmse(M, n, sigma, poldeg)

# Extract x and f(x) values from each simulation
x_values <- round(results$xmat, 7)
f_values <- round(results$f_values, 7)
y_values <- round(results$ymat, 7)
# Create a matrix for (x, f(x)) values
if (show_all || n <= 5) {
  x_fx_matrix <- rbind(x_values[1,], f_values[1,])
  colnames(x_fx_matrix) <- paste0("X", 1:n)
  rownames(x_fx_matrix) <- c("X", "f(X)")
  # Add y_simulation rows
  y_simulation_rows <- y_values[1:M,]
  row.names(y_simulation_rows) <- c(paste0("y_sim", 1:M))
} else {
  x_fx_matrix <- rbind(
    x_values[1, c(1:3, n)], 
    f_values[1, c(1:3, n)]
  )
  x_fx_matrix <- cbind(x_fx_matrix[,1:3], "...", x_fx_matrix[,4])
  colnames(x_fx_matrix) <- c(paste0("X", 1:3), "...", paste0("X", n))
  rownames(x_fx_matrix) <- c("X", "f(X)")
  # Add y_simulation rows
  y_simulation_rows <- rbind(
    y_values[1, c(1:3, n)], 
    y_values[2, c(1:3, n)], 
    y_values[3, c(1:3, n)], 
    rep("...", 5), 
    y_values[M, c(1:3, n)]
  )
  y_simulation_rows <- cbind(y_simulation_rows[,1:3], "...", y_simulation_rows[,4])
  colnames(y_simulation_rows) <- c(paste0("X", 1:3), "...", paste0("X", n))
  rownames(y_simulation_rows) <- c("y_sim1", "y_sim2", "y_sim3", "...", 
                                   paste0("y_sim", M))
}
# Combine matrices for output
output_matrix <- rbind(x_fx_matrix, y_simulation_rows)
# Print the (x, f(x), y) matrix
knitr::kable(output_matrix, align = "c")
```



|        |     X1     |     X2     |     X3     | ... |    X100     |
|:-------|:----------:|:----------:|:----------:|:---:|:-----------:|
|X       | -0.5604756 | -0.2301775 | 1.5587083  | ... | -1.0264209  |
|f(X)    | -1.2774088 | 0.3970961  | 10.6192538 | ... | -6.4040475  |
|y_sim1  | -6.9606613 | 2.4521658  | 8.6457188  | ... | -15.8878882 |
|y_sim2  | 16.313074  | 10.8963999 | 8.4980934  | ... |  3.595269   |
|y_sim3  | -6.9993463 | -5.6244156 | 3.1109442  | ... | -9.2403868  |
|...     |    ...     |    ...     |    ...     | ... |     ...     |
|y_sim50 | -5.2307998 | 9.4178438  | 1.4436574  | ... | -1.8972215  |
<!-- for latex use knitr::kable(output_matrix, format = "latex", booktabs = TRUE, align = "c")-->

After obtaining 50 training datasets, we will discuss how each value of MSPE and its components is computed in the next program snippet. Let's continue to Using  $(X,Y_{sim1})$ data, `simmse`function estimate cubic polynomial function, find and store the estimated coefficients , then calculate \(\hat{f}_3(x) = \hat{\beta}_0 + \hat{\beta}_1 x + \hat{\beta}_2 x^2 + \hat{\beta}_3 x^3\) value  using \(x_{1}\) in simulation 1 data ,\(X_{1}\), then in simulation 1 data ,\(x_{2}\) in simulation 1 data ,\(X_{1}\), upto x100. save it as a row in simulation 1 data ,\(s_{1}\), which is named as in simulation 1 data ,\(y_{sim1}\) in the output table. This is done for all 50 training sample data. Thus, We have a data frame like following

$$
\begin{array}{c|cccc}
& x_1 & x_2 & \cdots & x_{100} \\
\hline
s_1 & \hat{f}(x_1) & \hat{f}(x_2) & \cdots & \hat{f}(x_{100}) \\
s_2 & \hat{f}(x_1) & \hat{f}(x_2) & \cdots & \hat{f}(x_{100}) \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
s_{50} & \hat{f}(x_1) & \hat{f}(x_2) & \cdots & \hat{f}(x_{100}) \\
\end{array}
$$
Then, we calculate the expected value of the predicted \( \hat{f} \) for all \( x \) values. Next, we determine the variance of the predicted \( \hat{f} \) values for all \( x \) values. We also calculate the squared bias for all \( x \) values. To present the results, we create a data frame that includes the predicted values for each simulation, the sum of the predicted values, the expected predicted values, the variance, and the squared bias for the selected \( x \) values to calculate the MSPE and its components. For better readability, we add a column with ellipses ("...") to indicate the continuation of values between \( x_3 \) and \( x_n \). We then print the data frame to display the results.


``` r
# Calculate expected fhat for all x values
expected_fhat_all <- apply(results$yhat, 2, mean)
# Calculate sum of fhats for all x values
sum_fhat_all <- colSums(results$yhat)
# Calculate variance of fhat for all x values
variance_fhat_all <- apply(results$yhat, 2, 
                function(x) var(x) * (length(x) - 1) / length(x))
# Calculate squared bias for all x values
squared_bias_all <- (expected_fhat_all - results$f_values[1,])^2
# Select specific x values (x1, x2, x3, xn)
selected_indices <- c(1, 2, 3, n)
# Extract the specific values for display
expected_fhat <- expected_fhat_all[selected_indices]
sum_fhat <- sum_fhat_all[selected_indices]
variance_fhat <- variance_fhat_all[selected_indices]
squared_bias <- squared_bias_all[selected_indices]
# Add "..." column between X3 and Xn
add_dots <- function(mat) {
  dots <- matrix("...", nrow(mat), 1)
  cbind(mat[, 1:3], dots, mat[, 4])
}
# Create a data frame for output
output_matrix <- rbind(results$yhat[, selected_indices], 
                       sum_fhat, expected_fhat, variance_fhat, squared_bias)
colnames(output_matrix) <- c("X1", "X2", "X3", paste0("X", n))
rownames(output_matrix) <- c(paste0("fhatsim_", 1:M), "Sum_fhat", 
                             "Expected_fhat", "Variance_fhat", "Squared_Bias")
# Add dots column
output_matrix <- add_dots(output_matrix)
# Update column names to include the last X column name
colnames(output_matrix) <- c("X1", "X2", "X3", "...", paste0("X", n))
# Convert the numeric part of the matrix to numeric and round it
numeric_part <- output_matrix[, c(1:3, 5)]
numeric_part <- apply(numeric_part, 2, as.numeric)
numeric_part <- round(numeric_part, 5)
# Combine the numeric part with the "..." column
output_matrix <- cbind(numeric_part[, 1:3], "..." = output_matrix[, 4], 
                       numeric_part[, 4])
# If-else structure for printing
if (show_all || n <= 5) {
  print(output_matrix, quote = FALSE)
} else {
  output_matrix_part <- rbind(
    output_matrix[1:3, ],
    rep("...", ncol(output_matrix)),
    output_matrix[M, ],
    output_matrix[(M+1):nrow(output_matrix), ]
  )
rownames(output_matrix_part) <- c(paste0("Simulation_", 1:3), "...", 
paste0("Simulation_",M),rownames(output_matrix)[(M+1):nrow(output_matrix)])
knitr::kable(output_matrix_part, align = "c")
}
```



|              |    X1     |    X2    |    X3     | ... |            |
|:-------------|:---------:|:--------:|:---------:|:---:|:----------:|
|Simulation_1  | -1.41639  | 0.26594  |  8.23025  | ... |   -6.936   |
|Simulation_2  |  0.54238  | 1.29428  | 11.11243  | ... |  -3.2481   |
|Simulation_3  | -0.11851  | 0.92696  |  9.36313  | ... |  -5.12983  |
|...           |    ...    |   ...    |    ...    | ... |    ...     |
|Simulation_50 | -0.67598  | 1.03748  | 11.12986  | ... |  -5.94959  |
|Sum_fhat      | -53.77726 | 29.44154 | 526.14357 | ... | -319.57438 |
|Expected_fhat | -1.07555  | 0.58883  | 10.52287  | ... |  -6.39149  |
|Variance_fhat |  1.06094  | 0.66369  |  3.05423  | ... |  2.05494   |
|Squared_Bias  |  0.04075  | 0.03676  |  0.00929  | ... |  0.00016   |
<!-- for latex use knitr::kable(output_matrix_part, format = "latex", booktabs = TRUE, align = "c")-->

Formally, let \( \hat{f}_{X_1, i} \) represent the predicted \( \hat{f} \) value at \( X_1 \) for the \( i \)-th simulation. The expected \( \hat{f} \) value at \( X_1 \), denoted as \( \mathbb{E}[\hat{f}_{X_1}] \), is calculated as:

\begin{equation}
\text{Expected } \hat{f} \text{ at } X_1: \quad \mathbb{E}[\hat{f}_{X_1}] = \frac{1}{M} \sum_{i=1}^{M} \hat{f}_{X_1, i}
\end{equation}

Here is the output all 50 \(x_{1} \)s from all simulations (X1 column in simulation outtable above). Sum_hat is summation of these xs, and dividing this sum by 50 will give us Expected_fhat.


```
## fhat values at X1: -1.416392 0.5423814 -0.1185079 1.238812 -1.002696 -1.728647 -1.624869 0.05110822 0.07516055 0.0766192 -1.034525 -0.5050097 -1.735123 -0.5988579 -1.095943 -1.340411 -1.748233 -1.848451 -0.106242 -0.88527 -1.412619 -2.880681 -1.754998 -2.231159 0.8346244 -2.680983 -2.629536 -0.2490797 -0.4232302 -1.727644 0.03592634 -0.327935 -1.011624 0.07420027 -0.4940857 -1.317213 -3.845779 -0.264057 0.5208008 -2.416152 -1.251696 -2.112036 -2.561129 -1.392818 -1.065002 -1.547459 -1.362601 -1.980663 -0.8215615 -0.675977
```


```
## Expected fhat at X1: -1.075545
```

Formally, the variance of \( \hat{f} \) at \( X_1 \), denoted as \( \text{Var}(\hat{f}_{X_1}) \), is calculated by averaging the squared differences between each predicted \( \hat{f} \) value and the expected \( \hat{f} \) value at \( X_1 \):

\begin{equation}
\text{Var}(\hat{f}_{X_1}) = \frac{1}{M} \sum_{i=1}^{M} (\hat{f}_{X_1, i} - \mathbb{E}[\hat{f}_{X_1}])^2
\end{equation}

These squared differences provide the foundation for calculating the variance of \( \hat{f} \) at \( X_1 \). Displaying these values allows us to observe the individual contributions to the overall variance. For instance, first squared difference is \((-1.41639 - -1.07555)^2= 0.1161766\)

\begin{equation}
\text{Squared differences at } X_1: \quad (\hat{f}_{X_1, i} - \mathbb{E}[\hat{f}_{X_1}])^2
\end{equation}


```
## Squared differences: 0.1161766 2.617687 0.9159205 5.35625 0.005307006 0.4265418 0.3017563 1.269348 1.324124 1.327483 0.00168266 0.3255109 0.4350434 0.2272309 0.0004160689 0.07015411 0.4525082 0.5973825 0.9395488 0.03620467 0.1136189 3.258514 0.4616558 1.335443 3.648748 2.577429 2.414887 0.6830453 0.425515 0.4252324 1.235369 0.5589212 0.004085916 1.321915 0.3380952 0.05840347 7.674195 0.6585131 2.548321 1.797226 0.03102923 1.074312 2.20696 0.1006622 0.0001111626 0.2227026 0.08240117 0.8192387 0.06450774 0.1596548
```

This variance provides a measure of the dispersion of the model's predictions, indicating how much the predicted values vary from the expected value. Displaying the calculated variance allows us to verify the consistency of the model's predictions at \( X_1 \).

\begin{equation}
\text{Variance of } \hat{f} \text{ at } X_1: \quad \text{Var}(\hat{f}_{X_1})
\end{equation}


```
## Variance of fhat at X1: 1.06094
```

Next, we calculate the squared bias for the predicted \( \hat{f} \) at \( X_1 \). Formally, let \( \mathbb{E}[\hat{f}_{X_1}] \) be the expected predicted value at \( X_1 \) and \( f(X_1) \) be the true function value at \( X_1 \). The squared bias is given by:

\begin{equation}
\text{Squared bias at } X_1 = (\mathbb{E}[\hat{f}_{X_1}] - f(X_1))^2
\end{equation}

Firstly, we extract the true function value \( f(X_1) \) from the simulation results.

\begin{equation}
\text{True function value at } X_1: \quad f(X_1)
\end{equation}


```
## f(X1) values: -1.277409
```

The expected \( \hat{f} \) value at \( X_1 \), previously calculated as the mean of the predicted values across all simulations, is an essential part of the squared bias calculation.

\begin{equation}
\text{Expected } \hat{f} \text{ at } X_1: \quad \mathbb{E}[\hat{f}_{X_1}]
\end{equation}


```
## Expected fhat at X1: -1.075545
```

 Displaying the squared bias allows us to directly observe the magnitude of this error. This measure helps in understanding the bias component of the model's error at \( X_1 \).
 
\begin{equation}
\text{Squared bias at } X_1: \quad (\mathbb{E}[\hat{f}_{X_1}] - f(X_1))^2
\end{equation}


```
## Squared Bias at X1: 0.04074888
```

After calculating individual \( x \) values, we calculate aggregate measures for all \( x \) values for components of MSPE. As in this simulation we have 100 \( x \) values, we calculated 100 variance and squared bias.

Firstly, we calculate the expected variance of the predicted \( \hat{f} \) values for all \( x \) values. This is obtained by averaging the variances of the predicted \( \hat{f} \) values across all \( x \) values. This expected variance provides a summary measure of the dispersion of the model's predictions.

\begin{equation}
\text{Expected variance of } \hat{f} \text{ for all } x \text{ values} = \frac{1}{n} \sum_{j=1}^{n} \text{Var}(\hat{f}_{x_j})
\end{equation}

Secondly, we calculate the expected squared bias for all \( x \) values. This is done by averaging the squared biases across all \( x \) values. The expected squared bias gives an overall measure of the systematic error in the model's predictions.

\begin{equation}
\text{Expected squared bias of } \hat{f} \text{ for all } x \text{ values} = \frac{1}{n} \sum_{j=1}^{n} (\mathbb{E}[\hat{f}_{x_j}] - f(x_j))^2
\end{equation}

Lastly, we compute the irreducible error, which is the average of the irreducible errors across all simulations. The irreducible error represents the portion of the total error that cannot be reduced by the model, as it is inherent in the data.

\begin{equation}
\text{Irreducible error} = \frac{1}{M} \sum_{i=1}^{M} \text{Irrerr}_i
\end{equation}


```
## Expected Variance of fhat for polynomial degree 3 is: 2.172423
```

```
## Expected Squared Bias of fhat for polynomial degree 3 is: 0.1282653
```

```
## Irreducible error is: 63.23633
```

You can observe the same results as those in the row for $\text{PolyDeg3}$ from the main output table of the simulation, which presents results from $\text{PolyDeg1}$ to $\text{PolyDeg10}$. The main simulation performs this process for each polynomial degree in a loop. It is important to note that to decompose the MSPE as described above, we must use the true value of the function (as seen in the bias equation above). Since the true function is never known, we cannot calculate the components of the MSPE and obtain the MSPE by simply adding them. Instead, we must use the MSPE equation provided below. In our simulation, we calculated each MSPE for each training dataset and then found the expected value of these MSPEs. This approach is how MSPE is calculated with real datasets.

As we can calculate \((f(x_i)\) and \(\hat{f}(x_i))\) for each value , we have a data frame

$$
\begin{array}{c|cccc}
& x_1 & x_2 & \cdots & x_{100} \\
\hline
s_1 & \left[ f(x_1) - \hat{f}(x_1) \right]^2 & \left[ f(x_2) - \hat{f}(x_2) \right]^2 & \cdots & \left[ f(x_{100}) - \hat{f}(x_{100}) \right]^2 \\
s_2 & \left[ f(x_1) - \hat{f}(x_1) \right]^2 & \left[ f(x_2) - \hat{f}(x_2) \right]^2 & \cdots & \left[ f(x_{100}) - \hat{f}(x_{100}) \right]^2 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
s_{50} & \left[ f(x_1) - \hat{f}(x_1) \right]^2 & \left[ f(x_2) - \hat{f}(x_2) \right]^2 & \cdots & \left[ f(x_{100}) - \hat{f}(x_{100}) \right]^2 \\
\end{array}
$$

Then using this data frame, we can calculate MSPE for each simulation and then find the expected value of these MSPEs. This is how MSPE is calculated in real data sets.

For each simulation \( i \), the MSPE is computed as follows:

\begin{equation}
\text{MSPE}_i = \frac{1}{n} \sum_{j=1}^{n} (y_{ij} - \hat{y}_{ij})^2
\end{equation}
where \( y_{ij} \) represents the observed value and \( \hat{y}_{ij} \) represents the predicted value for the \( j \)-th observation in the \( i \)-th simulation.

We calculate the error for \( x_{i} \) in simulation 1 by subtracting the predicted value from the true value and squaring the result as seen in data frame. We then calculate the errors for all values of \( x \) in simulation 1 and find their average. This average represents the Mean Squared Prediction Error (MSPE) for \( x_{i} \) in simulation 1. This process is repeated across all 50 simulations. Below are the MSPE values for the first three simulations.



```
## Simulation 1 MSPE: 58.4078 
## Simulation 2 MSPE: 54.7409 
## Simulation 3 MSPE: 66.193
```

The expected MSPE is the average of the MSPE values across all simulations. Formally, the expected MSPE is calculated as follows:

\begin{equation}
\textbf{MSPE}=
\mathbb{E} \left[ (y_i - \hat{f}(x_i))^2 \right]
\end{equation}

where \(y_i\) is the observed value and \(\hat{f}(x_i)\) is the predicted value for the \(i\)-th observation.

This expected MSPE gives us a single value that summarizes the model's predictive performance over all simulations. It helps in understanding the average prediction error and serves as a benchmark for comparing different models or configurations.


```
## Expected MSPE: 60.9356
```

You can see these are the same results with those shown in the row for $\text{PolyDeg3}$ from the main output table of the simulation, which presents results from $\text{PolyDeg1}$ to $\text{PolyDeg10}$.

Before concluding this simulation, we will discuss its outcomes. From the main equation, it is evident that the Mean Squared Prediction Error (MSPE) can be decomposed into its components: bias, variance, and irreducible error. Ideally, adding these three components should yield the MSPE. However, this is not observed in the current simulation. For example, when the number of simulations, $M$, is increased to 500 and the number of $x$ in each training dataset, denoted as $n$, is raised to 100,000; the MSPE is 64.0103 with a bias squared of 0.0025, a variance of 0.0024, and an irreducible error of 64.0123. If we further increase $M$ to 2000, while maintaining $n$ at 100,000; the MSPE adjusts to 63.9974 with a bias squared of 0, a variance of 0.002, and an irreducible error of 64. 

These discrepancies arise due to the stochastic nature of the simulations and the finite sample sizes used. As $M$ and $n$ increase, the law of large numbers ensures that the estimated components (bias, variance, and irreducible error) converge to their true values. This convergence is why the calculated MSPE becomes more accurate and aligns more closely with the theoretical expectations as the number of simulations and the sample size increase. This trend is consistent across all polynomial degrees, illustrating the importance of large sample sizes and numerous simulations in achieving reliable statistical estimates.

## Simulation for Population Parameter
 
Although the variance-bias trade-off conceptually seems intuitive, at least from a mathematical standpoint, another practical question arises: Is it possible to observe the components of the decomposed Mean Squared Prediction Error (MSPE)? In real-world data, observing these components is challenging as we do not know the true function and irreducible error. However, we can illustrate this concept through simulations.

For our analysis, we revisit the example discussed earlier. We consider years of schooling, varying between 9 and 16 years, as our variable of interest. From this 'population', we repeatedly take random samples. The objective now is to utilize each sample to develop a predictor or a set of prediction rules. These rules aim to predict a number or a series of numbers (years of schooling in this simulation) that are also drawn from the same population. By doing so, we can effectively simulate and analyze the decomposition of the MSPE, providing a clearer understanding of its components and their interplay.



``` r
# Load necessary library
library(ggplot2)
M <- 5000 # Set the number of repeated samples
# Here is our population
populationX <- c(9,10,11,12,13,14,15,16)
# Generate repeated samples (M) with replacement
set.seed(123)
samples <- replicate(M, sample(populationX, 10, replace = TRUE))
samples <- t(samples)
head(samples)
```

```
##      [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10]
## [1,]   15   15   11   14   11   10   10   14   11    13
## [2,]   12   14   14    9   10   11   16   13   11    11
## [3,]    9   12    9    9   13   11   16   10   15    10
## [4,]    9   14   11   12   14    9   11   15   13    12
## [5,]   15   16   10   13   15    9    9   10   15    11
## [6,]   12   13   15   13   11   16   14    9   10    13
```

``` r
# Now, Let's use our predictors:
# Container to record all predictions
predictions <- matrix(0, nrow = M, ncol = 2)
colnames(predictions) <- c("fhat_1", "fhat_2")
# fhat_1 = 10
predictions[,1] <- 10
# fhat_2 - mean
predictions[,2] <- rowMeans(samples)
head(predictions)
```

```
##      fhat_1 fhat_2
## [1,]     10   12.4
## [2,]     10   12.1
## [3,]     10   11.4
## [4,]     10   12.0
## [5,]     10   12.3
## [6,]     10   12.6
```

``` r
# Now let's have our MSPE decomposition:
# MSPE
MSPE <- matrix(0, nrow = M, ncol = 2)
colnames(MSPE) <- c("fhat_1", "fhat_2")
for (i in 1:M) {
  MSPE[i,1] <- mean((populationX - predictions[i,1])^2)
  MSPE[i,2] <- mean((populationX - predictions[i,2])^2)
}
head(MSPE)
```

```
##      fhat_1 fhat_2
## [1,]   11.5   5.26
## [2,]   11.5   5.41
## [3,]   11.5   6.46
## [4,]   11.5   5.50
## [5,]   11.5   5.29
## [6,]   11.5   5.26
```

``` r
# Bias
bias <- colMeans(predictions) - mean(populationX)
bias_squared <- bias^2
# Variance (predictor)
var_fhat <- apply(predictions, 2, var)
# Variance (epsilon)
var_eps <- mean((populationX - mean(populationX))^2)
# Let's put them in a table:
VBtradeoff <- matrix(c(bias_squared, var_fhat, rep(var_eps, 2), 
                       colMeans(MSPE)), nrow = 2, byrow = FALSE)
rownames(VBtradeoff) <- c("fhat_1", "fhat_2")
colnames(VBtradeoff) <- c("Bias^2", "Var(fhat)", "Var(eps)", "MSPE")
round(VBtradeoff, 3) 
```

```
##        Bias^2 Var(fhat) Var(eps)   MSPE
## fhat_1   6.25     0.000     5.25 11.500
## fhat_2   0.00     0.539     5.25  5.788
```
  
This table clearly shows the decomposition of MSPE. The first column is the contribution to the MSPE from the bias, and the second column is the contribution from the variance of the predictor. These together make up the reducible error. The third column is the variance that comes from the data, the irreducible error. The last column is, of course, the total MSPE, and we can see that $\hat{f}_2$ is the better predictor because of its lower MSPE.[^ExtMspe] 

[^ExtMspe]: Other simulation examples can be explored by clicking the numbered links: [1](https://blog.zenggyu.com/en/post/2018-03-11/understanding-the-bias-variance-decomposition-with-a-simulated-experiment/), [2](https://www.r-bloggers.com/2019/06/simulating-the-bias-variance-tradeoff-in-r/), and [3](https://daviddalpiaz.github.io/r4sl/simulating-the-biasvariance-tradeoff.html).

## Biased estimator as a predictor

Up to this point, we have shown through simulation that a prediction function with zero bias but high variance often produces better predictions than a prediction function with zero variance but high bias. However, we can potentially obtain an even better prediction function that has some bias and some variance. A better prediction function means a smaller Mean Squared Prediction Error (MSPE). The key idea is that if the reduction in variance more than compensates for the increase in bias, then we have a better predictor.

To explore this, let's define a biased estimator of \(\mu_x\):

\begin{equation}
\hat{X}_{\text{biased}} = \hat{\mu}_x = \alpha \bar{X}
\end{equation}

where \(\bar{X}\) is the sample mean. The sample mean \(\bar{X}\) is an unbiased estimator of \(\mu_x\), and the parameter \(\alpha\) introduces bias. When \(\alpha\) is 1, \(\hat{\mu}_x\) becomes the unbiased sample mean.

The bias of the estimator \(\hat{\mu}_x\) is given by:

\begin{equation}
\operatorname{Bias}(\hat{\mu}_x) = \mathbb{E}[\hat{\mu}_x] - \mu_x = \alpha \mu_x - \mu_x = (\alpha - 1) \mu_x
\end{equation}

The variance of the estimator \(\hat{\mu}_x\) is:

\begin{equation}
\operatorname{Var}(\hat{\mu}_x) = \operatorname{Var}(\alpha \bar{X}) = \alpha^2 \operatorname{Var}(\bar{X})
\end{equation}

Since the variance of the sample mean \(\bar{X}\) is \(\frac{\sigma_{\varepsilon}^2}{n}\) (Check chapter 5.4), we have:

\begin{equation}
\operatorname{Var}(\hat{\mu}_x) = \alpha^2 \frac{\sigma_{\varepsilon}^2}{n}
\end{equation}

The MSPE and its bias-variance components:

\begin{equation}
\text{MSPE} = \mathbb{E}[(\hat{\mu}_x - \mu_x)^2] = \operatorname{Bias}^2(\hat{\mu}_x) + \operatorname{Var}(\hat{\mu}_x) + \sigma_{\varepsilon}^2
\end{equation}

where \(\sigma_{\varepsilon}^2\) is the irreducible error.

First, calculate the bias squared:

\begin{equation}
\operatorname{Bias}^2(\hat{\mu}_x) = [(\alpha - 1) \mu_x]^2 = (1 - \alpha)^2 \mu_x^2
\end{equation}

Next, calculate the variance:

\begin{equation}
\operatorname{Var}(\hat{\mu}_x) = \alpha^2 \frac{\sigma_{\varepsilon}^2}{n}
\end{equation}

Finally, the irreducible error is  \(\sigma_{\varepsilon}^2\). After combining these terms,we can shows that the MSPE for the biased estimator \(\hat{\mu}_x = \alpha \bar{X}\) is:

\begin{equation}
\text{MSPE} = (1 - \alpha)^2 \mu_x^2 + \alpha^2 \frac{\sigma_{\varepsilon}^2}{n} + \sigma_{\varepsilon}^2 = [(1-\alpha) \mu_x]^2 + \frac{1}{n} \alpha^2 \sigma_{\varepsilon}^2 + \sigma_{\varepsilon}^2
\end{equation}

The final expression for the MSPE of the biased estimator \(\hat{\mu}_x = \alpha \bar{X}\) combines the squared bias term \((1 - \alpha)^2 \mu_x^2\), the variance term \(\frac{\alpha^2 \sigma_{\varepsilon}^2}{n}\), and the irreducible error term \(\sigma_{\varepsilon}^2\). By adjusting \(\alpha\), we can balance the trade-off between bias and variance to minimize the MSPE. This highlights that a small amount of bias can be beneficial if it significantly reduces the variance, leading to a lower MSPE and thus a better predictor.

Our first observation is that when $\alpha$ is one, the bias will be zero.  Since it seems that MSPE is a convex function of $\alpha$, we can search for $\alpha$ that minimizes MSPE.  The value of \(\alpha\) that minimizes the MSPE is:

\begin{equation}
\frac{\partial \text{MSPE}}{\partial \alpha} =0 \rightarrow ~~ \alpha = \frac{\mu^2_x}{\mu^2_x+\sigma^2_\varepsilon/n}<1
\end{equation}

Check end of the chapter for step-by-step derivation of the optimal value of \(\alpha\) that minimizes the MSPE.

Using the same simulation sample above , lets calculate alpha and MSPE with this new biased prediction function, and compare all 3 MSPEs. 





``` r
pred <-rep(0, Ms)
# The magnitude of bias
alpha <- (mean(populationX))^2/((mean(populationX)^2+var_eps/10))
alpha
```

```
## [1] 0.9966513
```

``` r
# Biased predictor
for (i in 1:Ms) {
  pred[i] <- alpha*predictions[i,2]
}
# Check if E(alpha*Xbar) = alpha*mu_x
mean(pred)
```

```
## [1] 12.45708
```

``` r
alpha*mean(populationX)
```

```
## [1] 12.45814
```

``` r
# MSPE
MSPE_biased <- rep(0, Ms)
for (i in 1:Ms) {
  MSPE_biased[i] <- mean((populationX-pred[i])^2)
}
mean(MSPE_biased)
```

```
## [1] 5.786663
```
  
Let's add this predictor into our table:  


``` r
VBtradeoff <- matrix(0, 3, 4)
rownames(VBtradeoff) <- c("fhat_1", "fhat_2", "fhat_3")
colnames(VBtradeoff) <- c("Bias", "Var(fhat)", "Var(eps)", "MSPE")
VBtradeoff[1,1] <- bias1^2
VBtradeoff[2,1] <- bias2^2
VBtradeoff[3,1] <- (mean(populationX)-mean(pred))^2
VBtradeoff[1,2] <- var1
VBtradeoff[2,2] <- var2
VBtradeoff[3,2] <- var(pred)
VBtradeoff[1,3] <- var_eps
VBtradeoff[2,3] <- var_eps
VBtradeoff[3,3] <- var_eps
VBtradeoff[1,4] <- mean(MSPE[,1])
VBtradeoff[2,4] <- mean(MSPE[,2])
VBtradeoff[3,4] <- mean(MSPE_biased)
round(VBtradeoff, 3)
```

```
##         Bias Var(fhat) Var(eps)   MSPE
## fhat_1 6.250     0.000     5.25 11.500
## fhat_2 0.000     0.539     5.25  5.788
## fhat_3 0.002     0.535     5.25  5.787
```

As seen , increase in bias is lower than decrease in variance. The prediction function with some bias and variance is the best prediction function as it has the smallest MSPE. This example shows the difference between estimation and prediction for a simplest predictor, the mean of $X$.  

In the next chapter, we will explore overfitting and how it relates to and differs from the bias-variance tradeoff. 

**Note:** To find the value of \(\alpha\) that minimizes the Mean Squared Prediction Error (MSPE), we take the derivative of the MSPE with respect to \(\alpha\), set it to zero, and solve for \(\alpha\).

The MSPE is given by:

\begin{equation}
\text{MSPE} = [(1-\alpha) \mu_x]^2 + \frac{1}{n} \alpha^2 \sigma_{\varepsilon}^2 + \sigma_{\varepsilon}^2
\end{equation}

First, we take the derivative of MSPE with respect to \(\alpha\):

\begin{equation}
\frac{\partial \text{MSPE}}{\partial \alpha} = \frac{\partial}{\partial \alpha} \left[(1-\alpha)^2 \mu_x^2 + \frac{1}{n} \alpha^2 \sigma_{\varepsilon}^2 + \sigma_{\varepsilon}^2\right]
\end{equation}

We compute the derivative term by term:

\begin{equation}
\frac{\partial \text{MSPE}}{\partial \alpha} = 2(1-\alpha)(-1) \mu_x^2 + 2 \frac{1}{n} \alpha \sigma_{\varepsilon}^2
\end{equation}

Simplifying, we get:

\begin{equation}
\frac{\partial \text{MSPE}}{\partial \alpha} = -2(1-\alpha) \mu_x^2 + 2 \frac{1}{n} \alpha \sigma_{\varepsilon}^2
\end{equation}

To find the minimum MSPE, we set the derivative equal to zero:

\begin{equation}
-2(1-\alpha) \mu_x^2 + 2 \frac{1}{n} \alpha \sigma_{\varepsilon}^2 = 0
\end{equation}

Solving for \(\alpha\):

\begin{equation}
-2 \mu_x^2 + 2 \alpha \mu_x^2 + 2 \frac{1}{n} \alpha \sigma_{\varepsilon}^2 = 0
\end{equation}

\begin{equation}
- \mu_x^2 + \alpha \mu_x^2 + \frac{1}{n} \alpha \sigma_{\varepsilon}^2 = 0
\end{equation}

\begin{equation}
- \mu_x^2 + \alpha \left( \mu_x^2 + \frac{\sigma_{\varepsilon}^2}{n} \right) = 0
\end{equation}

\begin{equation}
\alpha \left( \mu_x^2 + \frac{\sigma_{\varepsilon}^2}{n} \right) = \mu_x^2
\end{equation}

\begin{equation}
\alpha = \frac{\mu_x^2}{\mu_x^2 + \frac{\sigma_{\varepsilon}^2}{n}}
\end{equation}

Thus, the value of \(\alpha\) that minimizes the MSPE is:

\begin{equation}
\alpha = \frac{\mu_x^2}{\mu_x^2 + \frac{\sigma_{\varepsilon}^2}{n}}
\end{equation}


 
