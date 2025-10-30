// Model 1: Hierarchical Logistic Regression with Sample Size Covariate
// Tests whether log(sample size) explains heterogeneity in success rates

data {
  int<lower=1> J;                    // number of groups
  array[J] int<lower=0> n;           // number of trials per group
  array[J] int<lower=0> r;           // number of successes per group
  vector[J] log_n_centered;          // log(n) centered by mean
}

parameters {
  real beta_0;                       // intercept (population mean)
  real beta_1;                       // slope for log(sample size)
  real<lower=0> tau;                 // residual between-group SD
  vector[J] alpha_raw;               // non-centered parameterization
}

transformed parameters {
  vector[J] alpha;                   // group effects (logit scale)
  vector[J] mu;                      // regression predictor
  vector[J] p;                       // success probabilities

  // Regression structure
  mu = beta_0 + beta_1 * log_n_centered;

  // Non-centered parameterization for better sampling
  alpha = mu + tau * alpha_raw;

  // Inverse logit transformation
  p = inv_logit(alpha);
}

model {
  // Priors
  beta_0 ~ normal(-2.6, 1.0);        // Weakly informative (centers on observed data)
  beta_1 ~ normal(0, 0.5);           // Allows substantial effects
  tau ~ normal(0, 0.5);              // Half-normal for scale
  alpha_raw ~ std_normal();          // Standard normal for NC parameterization

  // Likelihood
  r ~ binomial(n, p);
}

generated quantities {
  // Log-likelihood for LOO-CV
  vector[J] log_lik;

  // Posterior predictive samples
  array[J] int r_rep;

  // R-squared (variance explained by covariate)
  real<lower=0, upper=1> R2;

  // Effect size: how much does doubling sample size change success rate?
  real effect_double_n;

  // Compute log-likelihood and posterior predictive
  for (j in 1:J) {
    log_lik[j] = binomial_lpmf(r[j] | n[j], p[j]);
    r_rep[j] = binomial_rng(n[j], p[j]);
  }

  // R-squared: proportion of variance explained by regression vs random effects
  {
    real var_fitted = variance(mu);
    real var_total = var_fitted + tau^2;
    R2 = var_fitted / var_total;
  }

  // Effect of doubling sample size (on probability scale)
  effect_double_n = inv_logit(beta_0 + beta_1 * log(2.0)) - inv_logit(beta_0);
}
