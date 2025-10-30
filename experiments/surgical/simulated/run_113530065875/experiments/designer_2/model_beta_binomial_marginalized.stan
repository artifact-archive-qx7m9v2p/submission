// Beta-Binomial Model (Marginalized over theta)
// Designer #2 - Computational Efficiency Version
// Purpose: Fast inference by marginalizing out group-level parameters

data {
  int<lower=1> J;              // Number of groups (12)
  int<lower=0> n[J];           // Trials per group
  int<lower=0> r[J];           // Successes per group
}

parameters {
  real<lower=0> alpha;         // Beta shape parameter 1
  real<lower=0> beta;          // Beta shape parameter 2
}

model {
  // Priors on Beta shape parameters
  alpha ~ gamma(2, 0.2);       // Mean = 10, SD = 7.07
  beta ~ gamma(2, 0.02);       // Mean = 100, SD = 70.7

  // Marginalized likelihood (no theta parameters)
  for (j in 1:J) {
    r[j] ~ beta_binomial(n[j], alpha, beta);
  }
}

generated quantities {
  // Population-level summaries
  real mu_pop = alpha / (alpha + beta);
  real sigma_pop = sqrt(alpha * beta / ((alpha + beta)^2 * (alpha + beta + 1)));
  real kappa = alpha + beta;

  // Posterior predictive (must sample theta)
  vector[J] theta;
  vector[J] p;
  vector[J] log_lik;
  vector[J] r_rep;

  for (j in 1:J) {
    // Sample theta from posterior given r[j], n[j]
    theta[j] = beta_rng(alpha + r[j], beta + n[j] - r[j]);
    p[j] = theta[j];

    // Log-likelihood using beta-binomial
    log_lik[j] = beta_binomial_lpmf(r[j] | n[j], alpha, beta);

    // Posterior predictive
    r_rep[j] = beta_binomial_rng(n[j], alpha, beta);
  }

  // Overdispersion metric
  real expected_var = mu_pop * (1 - mu_pop);
  real actual_var = variance(theta);
  real phi = actual_var / expected_var;
}
