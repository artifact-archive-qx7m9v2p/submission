// Model 3: Exponential Saturation Model
// Functional form: Y = Y_max - (Y_max - Y_0) * exp(-r * x)
// Parameters: Y_max (asymptote), Y_0 (initial value), r (rate), sigma (residual SD)

data {
  int<lower=0> N;           // Number of observations
  vector[N] x;              // Predictor values
  vector[N] Y;              // Response values
}

parameters {
  real<lower=0> Y_max;      // Asymptotic maximum
  real<lower=0> delta;      // Y_max - Y_0 (total change)
  real<lower=0> r;          // Rate constant
  real<lower=0> sigma;      // Residual standard deviation
}

transformed parameters {
  real Y_0 = Y_max - delta;  // Initial value at x=0
  vector[N] mu;               // Mean function

  for (i in 1:N) {
    mu[i] = Y_max - delta * exp(-r * x[i]);
  }
}

model {
  // Priors (weakly informative)
  Y_max ~ normal(2.6, 0.3);    // Centered at observed max
  delta ~ normal(0.9, 0.3);    // Expected total change: ~2.6 - 1.7 = 0.9
  r ~ exponential(0.5);        // Mean rate = 2, concentrates on positive values
  sigma ~ normal(0, 0.25);     // Half-normal via truncation

  // Likelihood
  Y ~ normal(mu, sigma);
}

generated quantities {
  // Posterior predictive samples
  vector[N] Y_rep;
  vector[N] log_lik;  // For LOO-CV

  for (i in 1:N) {
    Y_rep[i] = normal_rng(mu[i], sigma);
    log_lik[i] = normal_lpdf(Y[i] | mu[i], sigma);
  }

  // Derived quantities for interpretation
  real half_saturation_x = -log(0.5) / r;  // x where 50% of change complete
  real pct_95_saturation_x = -log(0.05) / r;  // x where 95% of change complete

  // Compare with Michaelis-Menten half-saturation
  // For MM: K is x where Y = Y_max/2
  // For Exp: half_saturation_x is where Y = (Y_0 + Y_max)/2
}
