// Hierarchical Logit-Normal Model for Binomial Data
// Non-centered parameterization to avoid funnel geometry

data {
  int<lower=1> J;                  // number of groups
  array[J] int<lower=1> n;        // number of trials per group
  array[J] int<lower=0> r;        // number of successes per group
}

parameters {
  real mu;                         // population mean (logit scale)
  real<lower=0> tau;              // between-group SD
  array[J] real theta_raw;        // standardized group effects
}

transformed parameters {
  array[J] real theta;            // group-level logit success rates
  for (j in 1:J) {
    theta[j] = mu + tau * theta_raw[j];
  }
}

model {
  // Hyperpriors
  mu ~ normal(-2.6, 1.0);
  tau ~ normal(0, 0.5);  // half-normal via constraint tau > 0

  // Group effects (non-centered)
  theta_raw ~ std_normal();

  // Likelihood
  for (j in 1:J) {
    r[j] ~ binomial_logit(n[j], theta[j]);
  }
}

generated quantities {
  // Success probabilities (for interpretation)
  array[J] real<lower=0, upper=1> p;

  // Log-likelihood (for LOO)
  array[J] real log_lik;

  // Posterior predictive samples
  array[J] int r_pred;

  for (j in 1:J) {
    p[j] = inv_logit(theta[j]);
    log_lik[j] = binomial_logit_lpmf(r[j] | n[j], theta[j]);
    r_pred[j] = binomial_rng(n[j], p[j]);
  }
}
