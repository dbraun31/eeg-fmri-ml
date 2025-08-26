rm(list=ls())
library(tidyverse)
library(glue)
library(MASS)

# ======= FUNCTION DEFS ============ #

# --- DEFINE TRUE FUNCTION --- #
f <- function(x) {
    (x - 2)^2 + sin(3*x)
}

f <- function(x) {
    sin(3 * x) + 0.5 * cos(5 * x) + x/5
}

# --- DEFINE KERNEL --- #
rbf <- function(x1, x2, sigma_f=1, l=1) {
    sigma_f^2 * exp(-0.5 * outer(x1, x2, FUN = function(a, b) (a-b)^2) / l^2)
}

# --- DEFINE ACQUISITION FUNCTION --- #
ucb <- function(x_test, mu, sigma, k=1, minimize=TRUE) {
    
    if (minimize) {
        z <-  mu - k * sigma
        x_best <- x_test[z == min(z)]
    } else {
        z <-  mu + k * sigma
        x_best <- x_test[z == max(z)]
    }
    return(x_best)
    
}

get_posterior <- function(x_train, x_test, y_train, sigma_n) {
    K_xx <- rbf(x_train, x_train)
    K_xs <- rbf(x_train, x_test)
    K_ss <- rbf(x_test, x_test)
    K_xx_noise <- K_xx + sigma_n^2 * diag(length(x_train))
    
    mu <- as.vector(t(K_xs) %*% solve(K_xx_noise, y_train))
    Sigma <- K_ss - t(K_xs) %*% solve(K_xx_noise, K_xs)
    return(list(mu=mu, Sigma=Sigma))
}

# --- DEFINE INITIAL OBSERVATIONS --- #
# Observations
x_train <- c(-1, -0.5)
y_train <- f(x_train)

# Test range
x_test <- seq(-1, 3, length.out=100)

# --- DEFINE HYPERPARAMETERS --- #
# Observation noise
sigma_n <- 0.0001
# Input scale
l <- .4
# Output scale
sigma_f <- 1.5

result <- data.frame()
for (step in 1:10) {
    # --- OBTAIN POSTERIOR --- #
    post <- get_posterior(x_train, x_test, y_train, sigma_n)
    mu <- post$mu
    Sigma <- post$Sigma
    sigma <- sqrt(diag(Sigma))
    
    # Sample posterior
    f_sample <- mvrnorm(1000, mu, Sigma)
    
    # --- NEXT SAMPLE --- #
    x_next <- ucb(x_test, mu, sigma, k=500)
    y_next <- f(x_next)
    x_train <- c(x_train, x_next)
    y_train <- c(y_train, y_next)
    
    # Save result
    cis <- t(apply(f_sample, MARGIN=2, FUN = function(x) quantile(x, probs = c(.025, .975))))
    row  <- data.frame(step=step, x = x_test, x_next = x_next, mu=mu, sigma=sigma, ci_l=cis[,1], ci_h=cis[,2])
    result <- rbind(result, row)
    
}

true_f <- data.frame(x = x_test, fx = f(x_test))


result %>%
    mutate(step_lab = glue('Step {step}')) %>% 
    mutate(step_lab = factor(step_lab, levels = glue("Step {sort(unique(step))}"))) %>% 
    #filter(step == 1) %>%  
    ggplot(aes(x = x, y = mu)) + 
    geom_ribbon(aes(ymin = mu - sigma, ymax = mu + sigma), alpha = .7) +
    geom_ribbon(aes(ymin = ci_l, ymax = ci_h), alpha = .3) +
    geom_line(color = 'blue') +
    geom_point(aes(x = x_next), y = 2, size = 3, color = 'green') +
    geom_line(data=true_f, aes(x = x, y = fx), color = 'red') + 
    facet_wrap(~step_lab) + 
    theme_bw()

# so cool
