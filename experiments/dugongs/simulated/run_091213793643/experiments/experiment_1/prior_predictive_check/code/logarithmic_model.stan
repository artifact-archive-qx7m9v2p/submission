// Logarithmic Regression Model: Y = alpha + beta * log(x) + epsilon
// Prior Predictive Check and Full Inference Model

data {
  int<lower=0> N;              // Number of observations
  vector<lower=0>[N] x;        // Predictor (must be positive for log transform)
  array[N] real Y;             // Response variable
  int<lower=0,upper=1> prior_only;  // Flag: 1=prior predictive, 0=full inference
}

transformed data {
  vector[N] log_x = log(x);    // Pre-compute log(x) for efficiency
}

parameters {
  real alpha;                  // Intercept (Y when x=1, since log(1)=0)
  real beta;                   // Logarithmic slope
  real<lower=0> sigma;         // Residual standard deviation
}

model {
  // Priors (weakly informative, centered on EDA estimates)
  alpha ~ normal(1.75, 0.5);     // Center at EDA intercept
  beta ~ normal(0.27, 0.15);     // Center at EDA slope, allows negative
  sigma ~ normal(0, 0.2);        // Half-normal via lower bound

  // Likelihood (only evaluated if prior_only=0)
  if (!prior_only) {
    Y ~ normal(alpha + beta * log_x, sigma);
  }
}

generated quantities {
  // Predictions and log-likelihood for posterior predictive checks and LOO-CV
  array[N] real Y_pred;        // Point predictions (posterior mean)
  array[N] real Y_rep;         // Replicated data (includes noise)
  array[N] real log_lik;       // Log-likelihood for each observation (for LOO)

  // Generate predictions and compute log-likelihood
  for (n in 1:N) {
    real mu_n = alpha + beta * log_x[n];
    Y_pred[n] = mu_n;
    Y_rep[n] = normal_rng(mu_n, sigma);

    // Log-likelihood (only meaningful for posterior, not prior predictive)
    if (!prior_only) {
      log_lik[n] = normal_lpdf(Y[n] | mu_n, sigma);
    } else {
      log_lik[n] = 0;  // Dummy value for prior predictive
    }
  }
}
