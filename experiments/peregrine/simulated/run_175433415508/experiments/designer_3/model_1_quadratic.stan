// Model 1: Quadratic Negative Binomial with AR(1)
// Designer 3, Model D3M1_quadratic_nb_ar1
// Handles acceleration in growth via polynomial term

data {
  int<lower=0> N;                  // Number of observations
  vector[N] year;                  // Standardized year predictor
  array[N] int<lower=0> C;         // Count response
}

transformed data {
  vector[N] year_sq;
  year_sq = year .* year;          // Pre-compute quadratic term
}

parameters {
  // Regression coefficients
  real beta_0;                     // Intercept
  real beta_1;                     // Linear term
  real beta_2;                     // Quadratic term (KEY parameter)

  // Dispersion
  real<lower=0> phi;               // NB dispersion (inverse)

  // AR(1) structure
  real<lower=-1, upper=1> rho;     // Autocorrelation
  real<lower=0> sigma_ar;          // Innovation SD
  vector[N] z;                     // Standard normal innovations
}

transformed parameters {
  vector[N] epsilon;               // AR(1) errors
  vector[N] log_mu;                // Linear predictor

  // Non-centered AR(1) parameterization for better sampling
  epsilon[1] = sigma_ar * z[1] / sqrt(1 - rho^2);  // Stationary initialization
  for (t in 2:N) {
    epsilon[t] = rho * epsilon[t-1] + sigma_ar * z[t];
  }

  // Quadratic mean function
  log_mu = beta_0 + beta_1 * year + beta_2 * year_sq + epsilon;
}

model {
  // Priors
  beta_0 ~ normal(4.7, 0.5);       // Centered at log(109) from EDA
  beta_1 ~ normal(1.0, 0.5);       // Expected positive growth
  beta_2 ~ normal(0, 0.3);         // Weakly informative, allows curvature

  phi ~ gamma(2, 0.1);             // Allows severe overdispersion
  rho ~ beta(20, 2);               // Strong prior for high autocorrelation
  sigma_ar ~ normal(0, 0.5);       // Innovation noise

  z ~ std_normal();                // Non-centered parameterization

  // Likelihood
  C ~ neg_binomial_2_log(log_mu, phi);
}

generated quantities {
  // For model comparison and diagnostics
  vector[N] log_lik;               // Pointwise log-likelihood for LOO
  array[N] int C_rep;              // Posterior predictive replicates
  vector[N] mu;                    // Expected counts

  // Derived quantities
  real growth_at_start;            // Growth rate at year = -1.67
  real growth_at_center;           // Growth rate at year = 0
  real growth_at_end;              // Growth rate at year = +1.67
  real curvature;                  // Second derivative (acceleration)

  // Compute log-likelihood and predictions
  mu = exp(log_mu);
  for (t in 1:N) {
    log_lik[t] = neg_binomial_2_log_lpmf(C[t] | log_mu[t], phi);
    C_rep[t] = neg_binomial_2_rng(mu[t], phi);
  }

  // Growth rate = d(log_mu)/d(year) = beta_1 + 2*beta_2*year
  growth_at_start = beta_1 + 2 * beta_2 * (-1.67);
  growth_at_center = beta_1;
  growth_at_end = beta_1 + 2 * beta_2 * 1.67;

  // Curvature = second derivative
  curvature = 2 * beta_2;
}
