// Prior Predictive Check for Robust Logarithmic Regression
// This model samples from priors and generates predictions without conditioning on data

data {
  int<lower=0> N;           // Number of observations
  vector[N] x;              // Predictor values
}

generated quantities {
  // Sample from priors
  real alpha = normal_rng(2.0, 0.5);
  real beta = normal_rng(0.3, 0.3);
  real<lower=0> c = gamma_rng(2, 2);
  real<lower=0> nu = gamma_rng(2, 0.1);
  real<lower=0> sigma = fabs(cauchy_rng(0, 0.2));  // half-cauchy via absolute value

  // Generate predictions
  vector[N] mu;
  vector[N] y_sim;

  for (i in 1:N) {
    mu[i] = alpha + beta * log(x[i] + c);
    y_sim[i] = student_t_rng(nu, mu[i], sigma);
  }

  // Diagnostics: check for extreme values
  real min_y_sim = min(y_sim);
  real max_y_sim = max(y_sim);
  real mean_y_sim = mean(y_sim);
  real sd_y_sim = sd(y_sim);

  // Check monotonicity (difference between last and first)
  real monotonic_increase = mu[N] - mu[1];

  // Extrapolation check: predict at x=50
  real mu_x50 = alpha + beta * log(50 + c);
  real y_x50 = student_t_rng(nu, mu_x50, sigma);
}
