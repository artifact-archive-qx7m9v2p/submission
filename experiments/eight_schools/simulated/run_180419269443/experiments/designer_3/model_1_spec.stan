// Model 1: Weakly Informative Hierarchical Model
// Designer 3 - Bayesian Meta-Analysis
// Non-centered parameterization for computational efficiency

data {
  int<lower=1> J;               // Number of studies
  vector[J] y;                  // Observed effects
  vector<lower=0>[J] sigma;     // Known standard errors
}

parameters {
  real mu;                      // Mean effect (population level)
  real<lower=0> tau;            // Between-study SD
  vector[J] theta_raw;          // Non-centered study effects
}

transformed parameters {
  vector[J] theta;              // Centered study effects
  theta = mu + tau * theta_raw;
}

model {
  // Priors
  mu ~ normal(0, 25);           // Weakly informative: 95% mass on [-50, 50]
  tau ~ normal(0, 10);          // Half-normal (by constraint): median ~6.7

  // Non-centered parameterization
  theta_raw ~ std_normal();

  // Likelihood
  y ~ normal(theta, sigma);
}

generated quantities {
  // Derived quantities for reporting
  real I_squared;               // Heterogeneity measure
  real theta_new;               // Predictive distribution for new study
  vector[J] log_lik;            // For LOO-CV
  vector[J] y_rep;              // Posterior predictive replications

  // I-squared statistic
  {
    real sigma_pooled_sq = mean(sigma .* sigma);
    I_squared = tau^2 / (tau^2 + sigma_pooled_sq);
  }

  // Prediction for new study
  theta_new = normal_rng(mu, tau);

  // Log-likelihood for each observation (for LOO)
  for (j in 1:J) {
    log_lik[j] = normal_lpdf(y[j] | theta[j], sigma[j]);
  }

  // Posterior predictive replicates
  for (j in 1:J) {
    y_rep[j] = normal_rng(theta[j], sigma[j]);
  }
}
