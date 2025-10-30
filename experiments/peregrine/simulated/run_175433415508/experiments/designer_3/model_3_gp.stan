// Model 3: Gaussian Process Negative Binomial
// Designer 3, Model D3M3_gp_nb
// Non-parametric flexible mean function

data {
  int<lower=0> N;                  // Number of observations
  vector[N] year;                  // Standardized year predictor
  array[N] int<lower=0> C;         // Count response
}

transformed data {
  // Precompute squared distance matrix for efficiency
  matrix[N, N] dist_sq;
  for (i in 1:N) {
    for (j in 1:N) {
      dist_sq[i, j] = square(year[i] - year[j]);
    }
  }
}

parameters {
  // Mean function (linear trend)
  real beta_0;                     // Intercept
  real beta_1;                     // Linear trend

  // GP hyperparameters
  real<lower=0> alpha_gp;          // Marginal standard deviation (amplitude)
  real<lower=0> length_scale;      // Length scale (KEY parameter)

  // GP latent function (non-centered)
  vector[N] eta;                   // Standard normal innovations

  // Dispersion
  real<lower=0> phi;               // NB dispersion (inverse)
}

transformed parameters {
  vector[N] f;                     // GP realization
  vector[N] log_mu;                // Linear predictor

  {
    matrix[N, N] L_K;              // Cholesky factor of covariance
    matrix[N, N] K;                // Covariance matrix

    // Construct squared exponential covariance
    K = alpha_gp^2 * exp(-0.5 * dist_sq / length_scale^2);

    // Add jitter for numerical stability
    for (n in 1:N) {
      K[n, n] = K[n, n] + 1e-9;
    }

    // Cholesky decomposition
    L_K = cholesky_decompose(K);

    // Non-centered parameterization
    f = L_K * eta;
  }

  // Mean function = linear trend + GP deviation
  log_mu = beta_0 + beta_1 * year + f;
}

model {
  // Priors
  beta_0 ~ normal(4.7, 0.5);       // Intercept
  beta_1 ~ normal(1.0, 0.5);       // Linear trend

  // GP hyperparameters
  alpha_gp ~ normal(0, 1);         // Moderate variation on log scale
  length_scale ~ inv_gamma(3, 3);  // Mode around 1, encourages smoothness

  // Non-centered GP
  eta ~ std_normal();

  // Dispersion
  phi ~ gamma(2, 0.1);

  // Likelihood
  C ~ neg_binomial_2_log(log_mu, phi);
}

generated quantities {
  // For model comparison and diagnostics
  vector[N] log_lik;
  array[N] int C_rep;
  vector[N] mu;

  // Derived quantities
  real mean_trend_contribution;    // Average contribution of linear trend
  real mean_gp_contribution;       // Average contribution of GP
  real gp_amplitude;               // Actual amplitude of GP realization
  real correlation_decay_at_1SD;   // Correlation at 1 SD distance
  int<lower=0> num_inflections;    // Number of inflection points (approximate)

  // Compute quantities
  mu = exp(log_mu);
  for (t in 1:N) {
    log_lik[t] = neg_binomial_2_log_lpmf(C[t] | log_mu[t], phi);
    C_rep[t] = neg_binomial_2_rng(mu[t], phi);
  }

  // Decompose contributions
  mean_trend_contribution = mean(beta_1 * year);
  mean_gp_contribution = mean(f);
  gp_amplitude = max(f) - min(f);

  // Correlation at distance = 1 SD
  correlation_decay_at_1SD = alpha_gp^2 * exp(-0.5 / length_scale^2);

  // Count inflection points (where second derivative changes sign)
  // Approximate via sign changes in first differences of f
  num_inflections = 0;
  {
    vector[N-1] first_diff;
    vector[N-2] second_diff;

    for (t in 1:(N-1)) {
      first_diff[t] = f[t+1] - f[t];
    }

    for (t in 1:(N-2)) {
      second_diff[t] = first_diff[t+1] - first_diff[t];
    }

    for (t in 1:(N-3)) {
      if (second_diff[t] * second_diff[t+1] < 0) {
        num_inflections += 1;
      }
    }
  }
}
