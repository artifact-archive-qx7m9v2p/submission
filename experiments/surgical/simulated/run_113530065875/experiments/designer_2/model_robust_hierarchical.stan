// Robust Hierarchical Binomial Model with Student-t Hyperpriors
// Designer #2 - Alternative Parameterization
// Purpose: Handle outlier groups (2, 4, 8) without excessive shrinkage

data {
  int<lower=1> J;              // Number of groups (12)
  int<lower=0> n[J];           // Trials per group
  int<lower=0> r[J];           // Successes per group
}

parameters {
  real mu;                      // Population mean (logit scale)
  real<lower=0> tau;            // Between-group scale (logit scale)
  real<lower=2, upper=30> nu;   // Degrees of freedom for Student-t
  vector[J] theta;              // Group-level logit rates
}

model {
  // Priors
  mu ~ normal(-2.5, 1);         // Weakly informative, centered at ~8% success rate
  tau ~ cauchy(0, 1);           // Half-Cauchy for scale parameter
  nu ~ gamma(2, 0.1);           // Prior on degrees of freedom
                                 // Mean = 20, SD = 14.14, allows adaptation

  // Robust hierarchical structure (heavy-tailed hyperprior)
  theta ~ student_t(nu, mu, tau);

  // Likelihood
  r ~ binomial_logit(n, theta);
}

generated quantities {
  vector[J] p = inv_logit(theta);           // Success probabilities
  vector[J] log_lik;                         // For LOO-CV
  vector[J] r_rep;                           // Posterior predictive
  vector[J] z_score = (theta - mu) / tau;   // Standardized residuals

  real mu_prob = inv_logit(mu);              // Population mean probability

  // Posterior predictive and log-likelihood
  for (j in 1:J) {
    log_lik[j] = binomial_logit_lpmf(r[j] | n[j], theta[j]);
    r_rep[j] = binomial_rng(n[j], p[j]);
  }

  // Diagnostics
  real max_zscore = max(fabs(z_score));      // Maximum absolute z-score
  int<lower=0> n_outliers = sum(fabs(z_score) > 2.5);  // Count of outliers

  // Overdispersion metric
  real observed_var = variance(p);
  real expected_var = mu_prob * (1 - mu_prob) / 12.0;  // Approx. binomial variance
  real phi = observed_var / expected_var;    // Dispersion parameter
}
