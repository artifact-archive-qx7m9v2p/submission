// Quadratic Polynomial Model
// Y = alpha + beta1 * x + beta2 * x^2 + epsilon
// Hypothesis: Polynomial approximation (local fit only, not for extrapolation)

data {
  int<lower=0> N;               // Number of observations
  vector[N] x;                  // Predictor
  vector[N] Y;                  // Response
}

transformed data {
  vector[N] x2;
  x2 = x .* x;                  // Precompute x^2
}

parameters {
  real alpha;                   // Intercept (Y at x=0)
  real beta1;                   // Linear coefficient
  real beta2;                   // Quadratic coefficient
  real<lower=0> sigma;          // Residual standard deviation
}

transformed parameters {
  vector[N] mu;
  mu = alpha + beta1 * x + beta2 * x2;
}

model {
  // Priors (weakly informative, centered on EDA estimates)
  alpha ~ normal(1.7, 0.3);     // Intercept near lower Y range
  beta1 ~ normal(0.1, 0.05);    // Positive linear term
  beta2 ~ normal(-0.002, 0.001); // Negative quadratic for saturation
  sigma ~ normal(0, 0.15);       // Half-normal via constraint

  // Likelihood
  Y ~ normal(mu, sigma);
}

generated quantities {
  vector[N] y_rep;              // Posterior predictive samples
  vector[N] log_lik;            // Pointwise log-likelihood for LOO-CV

  // Interpretable quantities
  real vertex_x;                 // x value at parabola vertex (max/min)
  real vertex_Y;                 // Y value at vertex
  real discriminant;             // b^2 - 4ac (for checking if vertex exists)
  int<lower=0,upper=1> vertex_in_range;  // 1 if vertex in [0, max(x)]

  // For posterior predictive checks
  real mean_y_rep;
  real sd_y_rep;
  real min_y_rep;
  real max_y_rep;

  // Vertex calculation: x = -beta1 / (2*beta2)
  // Only meaningful if beta2 != 0
  if (fabs(beta2) > 1e-10) {
    vertex_x = -beta1 / (2 * beta2);
    vertex_Y = alpha + beta1 * vertex_x + beta2 * vertex_x^2;
  } else {
    vertex_x = 999;  // Placeholder for "no vertex" (linear)
    vertex_Y = 999;
  }

  discriminant = beta1^2 - 4 * alpha * beta2;
  vertex_in_range = (vertex_x >= 0 && vertex_x <= max(x)) ? 1 : 0;

  for (i in 1:N) {
    y_rep[i] = normal_rng(mu[i], sigma);
    log_lik[i] = normal_lpdf(Y[i] | mu[i], sigma);
  }

  mean_y_rep = mean(y_rep);
  sd_y_rep = sd(y_rep);
  min_y_rep = min(y_rep);
  max_y_rep = max(y_rep);
}
