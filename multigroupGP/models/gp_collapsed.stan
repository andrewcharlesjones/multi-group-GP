data {
  int<lower=1> N;
  real x[N];
  vector[N] y;
}
transformed data {
  real delta = 1e-9;
}
parameters {

  // Covariance function parameters
  real<lower=0> outputvariance;
  real<lower=0> lengthscale;

  // Noise variance
  real<lower=0> sigma;

  // Intercepts
  real beta;

}
transformed parameters {

  // Covariance matrix
  cov_matrix[N] K = gp_exp_quad_cov(x, outputvariance, lengthscale);
  K = K + sigma * diag_matrix(rep_vector(1.0, N));

  // Chol. factor of K + D_tau
  cholesky_factor_cov[N] L_K = cholesky_decompose(K);
}
model {

  // Covariance function parameters
  outputvariance ~ normal(0, 1);
  lengthscale ~ inv_gamma(5, 5);

  // Noise variance
  sigma ~ inv_gamma(1, 1);

  // Intercepts
  beta ~ std_normal();

  // Data
  y ~ multi_normal_cholesky(rep_vector(beta, N), L_K);
}
generated quantities {
  vector[N] m;
  m = y - beta;
}



