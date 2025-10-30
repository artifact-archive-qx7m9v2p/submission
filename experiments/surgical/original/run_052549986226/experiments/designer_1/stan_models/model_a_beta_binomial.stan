// Model A: Homogeneous Beta-Binomial
// Standard parameterization with alpha and beta shape parameters
// Designer 1 - Beta-binomial baseline model

data {
  int<lower=1> N;                  // Number of groups
  array[N] int<lower=0> n_trials;  // Trials per group
  array[N] int<lower=0> r_success; // Successes per group
}

parameters {
  real<lower=0> alpha;             // Beta shape parameter 1
  real<lower=0> beta;              // Beta shape parameter 2
}

model {
  // Hyperpriors
  alpha ~ gamma(2, 0.5);
  beta ~ gamma(2, 0.1);

  // Likelihood (vectorized)
  r_success ~ beta_binomial(n_trials, alpha, beta);
}

generated quantities {
  // Derived quantities of interest
  real mean_p = alpha / (alpha + beta);
  real kappa = alpha + beta;
  real var_p = (alpha * beta) / ((alpha + beta)^2 * (alpha + beta + 1));

  // Overdispersion parameter (should be close to observed phi ~ 3.5)
  real phi = 1 + var_p / (mean_p * (1 - mean_p));

  // Posterior predictive for group-level rates
  array[N] real<lower=0, upper=1> p_rep;
  for (i in 1:N) {
    p_rep[i] = beta_rng(alpha, beta);
  }

  // Posterior predictive for data
  array[N] int r_rep;
  for (i in 1:N) {
    real p_i = beta_rng(alpha, beta);
    r_rep[i] = binomial_rng(n_trials[i], p_i);
  }

  // Log-likelihood for LOO-CV
  array[N] real log_lik;
  for (i in 1:N) {
    log_lik[i] = beta_binomial_lpmf(r_success[i] | n_trials[i], alpha, beta);
  }

  // Posterior mean estimates for each group (shrinkage estimators)
  array[N] real p_posterior_mean;
  for (i in 1:N) {
    // Posterior mean of p_i given observed data
    // For beta-binomial, this is: (r + alpha) / (n + alpha + beta)
    p_posterior_mean[i] = (r_success[i] + alpha) / (n_trials[i] + alpha + beta);
  }
}
