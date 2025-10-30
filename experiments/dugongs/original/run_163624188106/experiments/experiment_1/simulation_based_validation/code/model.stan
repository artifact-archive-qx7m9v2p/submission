data {
  int<lower=1> N;
  vector[N] x;
  vector<lower=0>[N] Y;
}

transformed data {
  vector[N] log_x = log(x);
  vector[N] log_Y = log(Y);
}

parameters {
  real alpha;                    // Log-scale intercept
  real beta;                     // Power law exponent
  real<lower=0> sigma;           // Log-scale residual SD
}

model {
  // Priors
  alpha ~ normal(0.6, 0.3);
  beta ~ normal(0.13, 0.1);
  sigma ~ normal(0, 0.1);        // Half-normal via constraint

  // Likelihood
  log_Y ~ normal(alpha + beta * log_x, sigma);
}

generated quantities {
  vector[N] log_lik;              // For LOO-CV
  vector[N] y_pred;               // Posterior predictive (original scale)
  vector[N] log_y_pred;           // Posterior predictive (log scale)
  real R_squared;                 // Bayesian R²

  // Log-likelihood for LOO
  for (i in 1:N) {
    log_lik[i] = normal_lpdf(log_Y[i] | alpha + beta * log_x[i], sigma);
  }

  // Posterior predictions (log scale)
  for (i in 1:N) {
    log_y_pred[i] = normal_rng(alpha + beta * log_x[i], sigma);
  }

  // Posterior predictions (original scale via exponential)
  y_pred = exp(log_y_pred);

  // Bayesian R² (variance explained in log scale)
  {
    real var_log_y = variance(log_Y);
    vector[N] mu = alpha + beta * log_x;
    real var_residuals = variance(log_Y - mu);
    R_squared = 1 - var_residuals / var_log_y;
  }
}
