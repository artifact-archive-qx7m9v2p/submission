// Model 3: Hierarchical Logistic Regression with Random Slopes
// Tests whether the effect of sample size varies across groups

data {
  int<lower=1> J;                    // number of groups
  array[J] int<lower=0> n;           // number of trials per group
  array[J] int<lower=0> r;           // number of successes per group
  vector[J] log_n_centered;          // log(n) centered by mean
}

parameters {
  real beta_0;                       // population mean intercept
  real beta_1;                       // population mean slope
  real<lower=0> tau_alpha;           // SD of intercepts
  real<lower=0> tau_gamma;           // SD of slopes
  vector[J] alpha_raw;               // non-centered intercepts
  vector[J] gamma_raw;               // non-centered slopes
  real<lower=-1, upper=1> rho;       // correlation between intercepts and slopes
}

transformed parameters {
  vector[J] alpha;                   // group intercepts (logit scale)
  vector[J] gamma;                   // group slopes
  vector[J] mu;                      // linear predictor
  vector[J] p;                       // success probabilities

  // Non-centered parameterization with correlation
  // Using Cholesky decomposition for efficiency
  {
    matrix[2, 2] L_Sigma;            // Cholesky factor of correlation matrix
    matrix[J, 2] z;                  // Standard normal draws
    matrix[J, 2] effects;            // Correlated effects

    // Build correlation matrix Cholesky factor
    L_Sigma[1, 1] = 1.0;
    L_Sigma[2, 1] = rho;
    L_Sigma[2, 2] = sqrt(1.0 - rho^2);
    L_Sigma[1, 2] = 0.0;

    // Stack raw parameters
    z[, 1] = alpha_raw;
    z[, 2] = gamma_raw;

    // Transform with correlation and scaling
    effects = z * L_Sigma';

    // Scale and shift
    alpha = beta_0 + tau_alpha * effects[, 1];
    gamma = beta_1 + tau_gamma * effects[, 2];
  }

  // Linear predictor with varying slopes
  for (j in 1:J) {
    mu[j] = alpha[j] + gamma[j] * log_n_centered[j];
  }

  // Inverse logit transformation
  p = inv_logit(mu);
}

model {
  // Priors
  beta_0 ~ normal(-2.6, 1.0);        // Population mean intercept
  beta_1 ~ normal(0, 0.5);           // Population mean slope
  tau_alpha ~ normal(0, 0.5);        // SD of intercepts
  tau_gamma ~ normal(0, 0.3);        // SD of slopes (more conservative)
  rho ~ uniform(-1, 1);              // Uninformative on correlation
  alpha_raw ~ std_normal();          // Standard normal
  gamma_raw ~ std_normal();          // Standard normal

  // Likelihood
  r ~ binomial(n, p);
}

generated quantities {
  // Log-likelihood for LOO-CV
  vector[J] log_lik;

  // Posterior predictive samples
  array[J] int r_rep;

  // Variance partitioning
  real var_intercepts;
  real var_slopes;
  real total_var;
  real prop_var_slopes;              // Proportion due to varying slopes

  // Test for slope variation
  int<lower=0, upper=1> slopes_vary; // 1 if tau_gamma > 0.1

  // Extreme slope detection
  real max_abs_slope;

  // Compute log-likelihood and posterior predictive
  for (j in 1:J) {
    log_lik[j] = binomial_lpmf(r[j] | n[j], p[j]);
    r_rep[j] = binomial_rng(n[j], p[j]);
  }

  // Variance components
  var_intercepts = tau_alpha^2;
  var_slopes = tau_gamma^2 * variance(log_n_centered);
  total_var = var_intercepts + var_slopes;
  prop_var_slopes = var_slopes / total_var;

  // Indicator for varying slopes (substantively meaningful)
  slopes_vary = (tau_gamma > 0.1) ? 1 : 0;

  // Detect extreme slope heterogeneity
  max_abs_slope = max(fabs(gamma));
}
