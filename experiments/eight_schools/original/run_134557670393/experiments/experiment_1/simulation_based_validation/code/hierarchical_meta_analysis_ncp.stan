// Bayesian Hierarchical Meta-Analysis Model
// Non-Centered Parameterization (for tau near 0)
//
// Likelihood: y_i ~ Normal(theta_i, sigma_i)
// Hierarchy: theta_i = mu + tau * theta_raw_i, theta_raw_i ~ Normal(0, 1)
// Priors: mu ~ Normal(0, 50), tau ~ Half-Cauchy(0, 5)

data {
  int<lower=1> J;                    // Number of studies
  vector[J] y;                       // Observed effect sizes
  vector<lower=0>[J] sigma;          // Known standard errors
}

parameters {
  real mu;                           // Population mean effect
  real<lower=0> tau;                 // Between-study SD
  vector[J] theta_raw;               // Standardized study effects
}

transformed parameters {
  vector[J] theta;                   // Study-specific effects
  theta = mu + tau * theta_raw;
}

model {
  // Priors
  mu ~ normal(0, 50);
  tau ~ cauchy(0, 5);
  theta_raw ~ normal(0, 1);          // Standard normal

  // Likelihood
  y ~ normal(theta, sigma);
}

generated quantities {
  // Log-likelihood for LOO-CV
  vector[J] log_lik;

  // Posterior predictive samples
  vector[J] y_rep;

  // Shrinkage assessment
  vector[J] shrinkage;

  for (j in 1:J) {
    log_lik[j] = normal_lpdf(y[j] | theta[j], sigma[j]);
    y_rep[j] = normal_rng(theta[j], sigma[j]);

    // Shrinkage towards mu: proportion of distance from y[j] to mu
    // that theta[j] has moved
    if (y[j] != mu) {
      shrinkage[j] = (y[j] - theta[j]) / (y[j] - mu);
    } else {
      shrinkage[j] = 0;
    }
  }
}
