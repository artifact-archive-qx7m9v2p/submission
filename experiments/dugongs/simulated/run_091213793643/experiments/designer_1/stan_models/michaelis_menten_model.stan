// Michaelis-Menten Saturation Model
// Y = Y_max - (Y_max - Y_min) * K / (K + x) + epsilon
// Hypothesis: True asymptotic saturation with finite upper limit

data {
  int<lower=0> N;               // Number of observations
  vector[N] x;                  // Predictor (must be positive)
  vector[N] Y;                  // Response
  real max_Y_obs;               // Maximum observed Y (for constraint)
}

parameters {
  real<lower=max_Y_obs> Y_max;  // Asymptotic maximum (must exceed observed max)
  real Y_min;                    // Baseline minimum
  real<lower=0> K;               // Half-saturation constant
  real<lower=0> sigma;           // Residual standard deviation
}

transformed parameters {
  vector[N] mu;                  // Mean function

  for (i in 1:N) {
    // MM equation: approaches Y_max as x -> infinity
    mu[i] = Y_max - (Y_max - Y_min) * K / (K + x[i]);
  }
}

model {
  // Priors (weakly informative)
  Y_max ~ normal(2.7, 0.3);     // Slightly above max(Y)=2.63
  Y_min ~ normal(1.5, 0.3);     // Near min(Y)=1.71, extrapolated to x=0
  K ~ normal(5, 3);              // Half-saturation around x=5 based on EDA
  sigma ~ normal(0, 0.2);        // Half-normal, same as log model

  // Likelihood
  Y ~ normal(mu, sigma);
}

generated quantities {
  vector[N] y_rep;               // Posterior predictive samples
  vector[N] log_lik;             // Pointwise log-likelihood for LOO-CV

  // Interpretable quantities
  real half_saturation_Y;        // Y value at half-saturation
  real pct_saturation_at_max_x;  // Percent saturation at max observed x
  real asymptote_range;          // Y_max - Y_min

  // For posterior predictive checks
  real mean_y_rep;
  real sd_y_rep;
  real min_y_rep;
  real max_y_rep;

  half_saturation_Y = (Y_max + Y_min) / 2;
  asymptote_range = Y_max - Y_min;
  pct_saturation_at_max_x = 100 * (1 - K / (K + max(x)));

  for (i in 1:N) {
    y_rep[i] = normal_rng(mu[i], sigma);
    log_lik[i] = normal_lpdf(Y[i] | mu[i], sigma);
  }

  mean_y_rep = mean(y_rep);
  sd_y_rep = sd(y_rep);
  min_y_rep = min(y_rep);
  max_y_rep = max(y_rep);
}
