/*
Hierarchical Binomial Model with Non-Centered Parameterization

This model uses a logit-normal hierarchy for binomial success rates.

Parameters:
  - mu: Population mean on logit scale
  - tau: Between-group SD on logit scale
  - theta_raw: Non-centered group deviations (standard normal)
  - theta: Group-level logit success rates (computed)

Priors:
  - mu ~ Normal(-2.5, 1)
  - tau ~ Half-Cauchy(0, 1)
  - theta_raw_j ~ Normal(0, 1)
  - theta_j = mu + tau * theta_raw_j

Likelihood:
  - r_j ~ Binomial(n_j, inv_logit(theta_j))
*/

data {
  int<lower=1> J;              // Number of groups
  array[J] int<lower=0> n;     // Sample sizes
  array[J] int<lower=0> r;     // Success counts
}

parameters {
  real mu;                     // Population mean (logit scale)
  real<lower=0> tau;           // Between-group SD (logit scale)
  vector[J] theta_raw;         // Non-centered group deviations
}

transformed parameters {
  vector[J] theta;             // Group-level logit success rates
  theta = mu + tau * theta_raw;
}

model {
  // Priors
  mu ~ normal(-2.5, 1);
  tau ~ cauchy(0, 1);
  theta_raw ~ std_normal();

  // Likelihood
  r ~ binomial_logit(n, theta);
}

generated quantities {
  // Posterior predictive checks
  array[J] int<lower=0> r_rep;
  vector[J] p;

  // Convert to probability scale
  p = inv_logit(theta);

  // Generate replicated data
  for (j in 1:J) {
    r_rep[j] = binomial_rng(n[j], p[j]);
  }
}
