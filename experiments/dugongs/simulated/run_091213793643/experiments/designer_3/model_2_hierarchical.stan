// Model 2: Hierarchical Replicate Model
// Structure: Y = Population Trend + Group Effects + Measurement Error
// Designer 3 - Hierarchical/Compositional Perspective

data {
  int<lower=1> N;                      // Total observations
  int<lower=1> J;                      // Number of unique x groups
  vector[N] Y;                         // Response
  int<lower=1,upper=J> group_id[N];   // Group membership for each observation
  vector[J] x_group;                   // Unique x values
  int<lower=0,upper=1> has_replicates[J];  // Indicator: 1 if group has replicates
}

transformed data {
  vector[J] log_x_group = log(x_group);
}

parameters {
  real alpha;
  real<lower=0> beta;
  real<lower=0> sigma_between;  // Between-group variability
  real<lower=0> sigma_within;   // Within-group variability
  vector[J] u_raw;              // Non-centered group effects
}

transformed parameters {
  vector[J] u;

  // Group effects: only for groups with replicates
  for (j in 1:J) {
    if (has_replicates[j] == 1) {
      u[j] = sigma_between * u_raw[j];
    } else {
      u[j] = 0;
    }
  }
}

model {
  vector[N] mu;

  // Priors
  alpha ~ normal(1.75, 0.5);
  beta ~ normal(0.27, 0.15);
  sigma_between ~ normal(0, 0.1);
  sigma_within ~ normal(0, 0.1);
  u_raw ~ std_normal();  // Non-centered parameterization

  // Mean model
  for (n in 1:N) {
    mu[n] = alpha + beta * log_x_group[group_id[n]] + u[group_id[n]];
  }

  // Likelihood
  Y ~ normal(mu, sigma_within);
}

generated quantities {
  vector[N] Y_rep;
  vector[N] log_lik;
  real<lower=0,upper=1> ICC;  // Intraclass correlation coefficient

  // ICC: proportion of variance between groups
  ICC = sigma_between^2 / (sigma_between^2 + sigma_within^2);

  for (n in 1:N) {
    real mu_n = alpha + beta * log_x_group[group_id[n]] + u[group_id[n]];
    Y_rep[n] = normal_rng(mu_n, sigma_within);
    log_lik[n] = normal_lpdf(Y[n] | mu_n, sigma_within);
  }
}
