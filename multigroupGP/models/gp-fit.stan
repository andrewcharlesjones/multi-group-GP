// Fit the hyperparameters of a Gaussian process with an 
// exponentiated quadratic kernel

data {
  int<lower=1> N;
  array[N] real x;
  vector[N] y;
}
parameters {
  real<lower=0> rho;
  real<lower=0> alpha;
  real<lower=0> sigma;
  real mu;
  vector[N] eta;
}
transformed parameters {
  matrix[N, N] L_K;
  matrix[N, N] K_noisy;
  matrix[N, N] K = gp_exp_quad_cov(x, alpha, rho);
  real sq_sigma = square(sigma);
  K_noisy = K + diag_matrix(rep_vector(sq_sigma, N));
  
  // diagonal elements
  //for (n in 1 : N) {
  //  K[n, n] = K[n, n] + sq_sigma;
  //}
  
  L_K = cholesky_decompose(K_noisy);
}
model {
  
  rho ~ inv_gamma(5, 5);
  alpha ~ inv_gamma(5, 5);
  sigma ~ normal(0, 1);
  mu ~ normal(0, 1);

  eta ~ std_normal();
  
  y ~ multi_normal_cholesky(rep_vector(mu, N), L_K);
}
generated quantities {
  vector[N] y_samples;
  y_samples = rep_vector(mu, N) + L_K * eta;
}

