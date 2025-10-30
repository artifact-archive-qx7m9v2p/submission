data {
  int<lower=0> N;              // Number of trials
  array[N] int<lower=0> n;     // Sample sizes
  array[N] int<lower=0> r;     // Number of successes
}

parameters {
  real mu_logit;               // Population mean on logit scale
  real<lower=0> sigma;         // Standard deviation on logit scale
  vector[N] eta;               // Standardized trial-specific effects (non-centered)
}

transformed parameters {
  vector[N] logit_theta;       // Trial-specific log-odds
  vector<lower=0, upper=1>[N] theta;  // Trial-specific probabilities

  logit_theta = mu_logit + sigma * eta;
  theta = inv_logit(logit_theta);
}

model {
  // Priors
  mu_logit ~ normal(-2.53, 1);
  sigma ~ normal(0, 1);        // Half-normal (constrained to sigma > 0)
  eta ~ std_normal();          // Standard normal for non-centered parameterization

  // Likelihood
  r ~ binomial_logit(n, logit_theta);
}

generated quantities {
  array[N] int<lower=0> r_rep;      // Posterior predictive samples
  array[N] real log_lik;             // Pointwise log-likelihood

  for (i in 1:N) {
    r_rep[i] = binomial_rng(n[i], theta[i]);
    log_lik[i] = binomial_logit_lpmf(r[i] | n[i], logit_theta[i]);
  }
}
