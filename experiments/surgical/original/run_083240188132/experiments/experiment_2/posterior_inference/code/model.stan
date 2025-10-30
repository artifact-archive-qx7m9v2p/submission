/*
 * Random Effects Logistic Regression Model
 *
 * Model:
 *   r_i | θ_i, n_i ~ Binomial(n_i, logit⁻¹(θ_i))  for i = 1, ..., N groups
 *   θ_i = μ + τ · z_i   (non-centered parameterization)
 *   z_i ~ Normal(0, 1)
 *
 * Priors:
 *   μ ~ Normal(-2.51, 1)     # logit(0.075)
 *   τ ~ HalfNormal(1)        # Between-group SD
 */

data {
  int<lower=1> N;              // Number of groups
  array[N] int<lower=0> n;     // Sample sizes
  array[N] int<lower=0> r;     // Event counts
}

parameters {
  real mu;                     // Population mean log-odds
  real<lower=0> tau;           // Between-group SD
  vector[N] z;                 // Standardized group effects
}

transformed parameters {
  vector[N] theta;             // Group-level log-odds
  vector[N] p;                 // Group-level probabilities

  theta = mu + tau * z;
  p = inv_logit(theta);
}

model {
  // Priors
  mu ~ normal(-2.51, 1.0);
  tau ~ normal(0, 1.0);       // HalfNormal via truncation
  z ~ std_normal();

  // Likelihood
  r ~ binomial_logit(n, theta);
}

generated quantities {
  // Log likelihood for each observation (for LOO-CV)
  vector[N] log_lik;

  // Posterior predictive samples
  array[N] int<lower=0> r_rep;

  // Derived quantities
  real pop_prob = inv_logit(mu);
  real icc = tau^2 / (tau^2 + pi()^2 / 3);

  for (i in 1:N) {
    log_lik[i] = binomial_logit_lpmf(r[i] | n[i], theta[i]);
    r_rep[i] = binomial_rng(n[i], p[i]);
  }
}
