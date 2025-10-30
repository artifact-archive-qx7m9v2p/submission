// Model 2: Power-Law with Saturation
// Functional form: Y = a + b * x^c, where 0 < c < 1
// Parameters: a (intercept), b (scale), c (exponent), sigma (residual SD)

data {
  int<lower=0> N;           // Number of observations
  vector[N] x;              // Predictor values
  vector[N] Y;              // Response values
}

parameters {
  real a;                   // Intercept at x=0
  real<lower=0> b;          // Scaling coefficient
  real<lower=0, upper=1> c; // Power exponent (0 < c < 1 for diminishing returns)
  real<lower=0> sigma;      // Residual standard deviation
}

transformed parameters {
  vector[N] mu;  // Mean function

  for (i in 1:N) {
    mu[i] = a + b * pow(x[i], c);
  }
}

model {
  // Priors (weakly informative)
  a ~ normal(1.7, 0.5);     // Intercept near Y at low x
  b ~ normal(0.5, 0.5);     // Positive scaling
  c ~ beta(2, 2);           // Symmetric on [0,1], mean=0.5
  sigma ~ normal(0, 0.25);  // Half-normal via truncation

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
  real elasticity_at_x10 = (b * c * pow(10.0, c-1)) * 10.0 / (a + b * pow(10.0, c));
  // Elasticity: (dY/dx) * (x/Y) at x=10

  // Classification of power-law type
  int is_log_like = c < 0.2 ? 1 : 0;      // Close to logarithmic if c near 0
  int is_linear_like = c > 0.8 ? 1 : 0;   // Close to linear if c near 1
}
