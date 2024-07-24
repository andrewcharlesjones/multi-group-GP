functions {
  matrix multigroup_rbf_kernel(matrix squared_dist_mat, int P, int[] groups, matrix group_dist_mat, real lengthscale, real outputvariance, real alpha, real delta) {
    int N = rows(squared_dist_mat);
    matrix[N, N] K;
    for (i in 1:(N - 1)) {
      K[i, i] = 1.0 + delta;
      for (j in (i + 1):N) {
        real group_diff_scaling = group_dist_mat[groups[i] + 1, groups[j] + 1];
        real scaling_term = 1 / pow(group_diff_scaling * square(alpha) + 1, P * 0.5);
        K[i, j] = scaling_term * exp(-0.5 * squared_dist_mat[i, j] * lengthscale / (group_diff_scaling * square(alpha) + 1));
        K[j, i] = K[i, j];
      }
    }
    K[N, N] = 1.0 + delta;
    return outputvariance * K;
  }
}
data {
  int<lower=1> N;
  int<lower=1> P;
  int<lower=1> ngroups;
  matrix[N, P] x;
  vector[N] y;
  matrix[N, ngroups] design;
  int groups[N];
  matrix[ngroups, ngroups] group_dist_mat;
  real<lower=0> lengthscale;
  real<lower=0> alpha;
}
transformed data {
  real delta = 1e-9;
  matrix[N, N] squared_dist_mat;
  for (i in 1:N) {
      squared_dist_mat[i, i] = 0;
      for (j in (i + 1):N) {
        squared_dist_mat[i, j] = sum(square(x[i] - x[j]));
        squared_dist_mat[j, i] = squared_dist_mat[i, j];
      }
  }
}
parameters {

  // Covariance function parameters
  real<lower=0> outputvariance;
  // real<lower=0> lengthscale;
  // real<lower=0> alpha;

  // Noise variance
  vector<lower=0>[ngroups] sigma;

  // Intercepts
  vector[ngroups] beta;

}
model {

  // Covariance matrix
  matrix[N, N] K = multigroup_rbf_kernel(squared_dist_mat, P, groups, group_dist_mat, lengthscale, outputvariance, alpha, delta);

  // Chol. factor of K + D_tau
  matrix[N, N] L_K = cholesky_decompose(K + diag_matrix(design * sigma));

  // Covariance function parameters
  outputvariance ~ inv_gamma(5, 5);
  // lengthscale ~ inv_gamma(5, 5);
  // alpha ~ inv_gamma(5, 5);

  // Noise variance
  sigma ~ inv_gamma(1, 1);

  // Intercepts
  beta ~ std_normal();

  // Data
  y ~ multi_normal_cholesky(design * beta, L_K);
}
//generated quantities {
//  vector[N] m;
//  vector[N] means;
//  m = y - design * beta;
//  means = design * beta;  
//}



