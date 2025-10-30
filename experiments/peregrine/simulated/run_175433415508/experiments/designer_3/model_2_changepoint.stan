// Model 2: Bayesian Changepoint Negative Binomial with AR(1)
// Designer 3, Model D3M2_changepoint_nb_ar1
// Explicit structural break with unknown location

data {
  int<lower=0> N;                  // Number of observations
  vector[N] year;                  // Standardized year predictor
  array[N] int<lower=0> C;         // Count response
}

parameters {
  // Regression coefficients
  real beta_0;                     // Intercept
  real beta_1;                     // Pre-changepoint slope
  real beta_2;                     // Slope change (KEY parameter)

  // Changepoint
  real<lower=-1.5, upper=1.5> tau; // Changepoint location (KEY parameter)

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

  // Non-centered AR(1) parameterization
  epsilon[1] = sigma_ar * z[1] / sqrt(1 - rho^2);
  for (t in 2:N) {
    epsilon[t] = rho * epsilon[t-1] + sigma_ar * z[t];
  }

  // Piecewise linear with continuous junction at tau
  for (t in 1:N) {
    if (year[t] <= tau) {
      log_mu[t] = beta_0 + beta_1 * year[t] + epsilon[t];
    } else {
      // After changepoint: slope becomes beta_1 + beta_2
      // Continuous at tau: must subtract beta_2 * tau to maintain continuity
      log_mu[t] = beta_0 + beta_1 * year[t] + beta_2 * (year[t] - tau) + epsilon[t];
    }
  }
}

model {
  // Priors
  beta_0 ~ normal(4.7, 0.5);       // Centered at log(109)
  beta_1 ~ normal(0.5, 0.5);       // Pre-change slope (slower growth)
  beta_2 ~ normal(0, 1.0);         // Slope change (can be positive or negative)

  tau ~ uniform(-1.5, 1.5);        // Weakly informative, avoids edges

  phi ~ gamma(2, 0.1);
  rho ~ beta(20, 2);
  sigma_ar ~ normal(0, 0.5);

  z ~ std_normal();

  // Likelihood
  C ~ neg_binomial_2_log(log_mu, phi);
}

generated quantities {
  // For model comparison and diagnostics
  vector[N] log_lik;
  array[N] int C_rep;
  vector[N] mu;

  // Derived quantities
  real slope_pre;                  // Slope before changepoint
  real slope_post;                 // Slope after changepoint
  real rate_change;                // Fold-change in growth rate
  int<lower=1, upper=N> cp_index;  // Changepoint index (observation number)
  real prop_before_cp;             // Proportion of data before changepoint

  // Compute quantities
  mu = exp(log_mu);
  for (t in 1:N) {
    log_lik[t] = neg_binomial_2_log_lpmf(C[t] | log_mu[t], phi);
    C_rep[t] = neg_binomial_2_rng(mu[t], phi);
  }

  // Slope interpretation
  slope_pre = beta_1;
  slope_post = beta_1 + beta_2;

  // Avoid division by zero
  if (fabs(slope_pre) > 0.001) {
    rate_change = slope_post / slope_pre;
  } else {
    rate_change = 999.0;  // Indicate undefined
  }

  // Find closest observation to changepoint
  {
    real min_dist = 999.0;
    cp_index = 1;
    for (t in 1:N) {
      real dist = fabs(year[t] - tau);
      if (dist < min_dist) {
        min_dist = dist;
        cp_index = t;
      }
    }
  }

  // Proportion before changepoint
  prop_before_cp = 0.0;
  for (t in 1:N) {
    if (year[t] <= tau) {
      prop_before_cp += 1.0 / N;
    }
  }
}
