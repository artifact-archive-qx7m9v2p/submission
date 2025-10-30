data {
  int<lower=0> N;              // Number of observations
  vector[N] log_x;             // Log-transformed predictor
  vector[N] log_Y;             // Log-transformed response
}

parameters {
  real alpha;                  // Intercept on log-log scale
  real beta;                   // Power law exponent (elasticity)
  real<lower=0> sigma;         // Residual SD on log scale
}

model {
  // Priors (REVISED based on prior predictive checks)
  alpha ~ normal(0.6, 0.3);          // Centered at log(1.8) ≈ 0.59
  beta ~ normal(0.12, 0.05);         // Centered at OLS estimate, tightened
  sigma ~ cauchy(0, 0.05);           // Half-Cauchy for log-scale residuals

  // Likelihood: log(Y) ~ Normal(α + β*log(x), σ)
  log_Y ~ normal(alpha + beta * log_x, sigma);
}

generated quantities {
  vector[N] log_lik;           // Log-likelihood for LOO-CV
  vector[N] y_rep;             // Posterior predictive samples (original scale)
  vector[N] mu_log;            // Mean on log scale

  for (i in 1:N) {
    mu_log[i] = alpha + beta * log_x[i];

    // Log-likelihood for each observation
    log_lik[i] = normal_lpdf(log_Y[i] | mu_log[i], sigma);

    // Posterior predictive samples (back-transformed to original scale)
    y_rep[i] = exp(normal_rng(mu_log[i], sigma));
  }
}
