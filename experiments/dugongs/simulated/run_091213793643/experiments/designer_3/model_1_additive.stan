// Model 1: Additive Decomposition Model
// Structure: Y = Parametric Trend + Gaussian Process + Noise
// Designer 3 - Hierarchical/Compositional Perspective

data {
  int<lower=1> N;
  vector[N] x;
  vector[N] Y;
}

transformed data {
  vector[N] log_x = log(x);
}

parameters {
  real alpha;
  real<lower=0> beta;  // Enforce positivity for saturation
  real<lower=0> sigma_noise;
  real<lower=0> eta;   // GP amplitude
  real<lower=0> rho;   // GP length scale
  vector[N] z_gp;      // Non-centered parameterization for GP
}

transformed parameters {
  vector[N] f_smooth;
  {
    matrix[N, N] L_K;
    matrix[N, N] K = gp_exp_quad_cov(x, eta, rho);

    // Add jitter for numerical stability
    for (n in 1:N) {
      K[n, n] = K[n, n] + 1e-9;
    }

    L_K = cholesky_decompose(K);
    f_smooth = L_K * z_gp;
  }
}

model {
  vector[N] mu = alpha + beta * log_x + f_smooth;

  // Priors
  alpha ~ normal(1.75, 0.5);
  beta ~ normal(0.27, 0.15);
  sigma_noise ~ normal(0, 0.15);
  eta ~ normal(0, 0.1);
  rho ~ inv_gamma(5, 5);
  z_gp ~ std_normal();  // Non-centered parameterization

  // Likelihood
  Y ~ normal(mu, sigma_noise);
}

generated quantities {
  vector[N] Y_rep;
  vector[N] log_lik;
  real<lower=0,upper=1> gp_proportion;  // Proportion of variance from GP

  // GP contribution to total variance
  gp_proportion = variance(f_smooth) / (variance(f_smooth) + sigma_noise^2);

  for (n in 1:N) {
    real mu_n = alpha + beta * log(x[n]) + f_smooth[n];
    Y_rep[n] = normal_rng(mu_n, sigma_noise);
    log_lik[n] = normal_lpdf(Y[n] | mu_n, sigma_noise);
  }
}
