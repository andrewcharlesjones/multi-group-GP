functions {
  matrix multigroup_rbf_kernel(matrix squared_dist_mat, int[] groups, matrix group_dist_mat, real lengthscale, real outputvariance, real alpha, real delta) {
    int N = rows(squared_dist_mat);
    matrix[N, N] K;
    for (i in 1:(N - 1)) {
      K[i, i] = 1.0 + delta;
      for (j in (i + 1):N) {
        real group_diff_scaling = group_dist_mat[groups[i] + 1, groups[j] + 1];
        real scaling_term = 1 / pow(group_diff_scaling * square(alpha) + 1, 0.5);
        K[i, j] = scaling_term * exp(-0.5 * squared_dist_mat[i, j] * lengthscale / (group_diff_scaling * square(alpha) + 1));
        K[j, i] = K[i, j];
      }
    }
    K[N, N] = 1.0 + delta;
    return outputvariance * K;
  }
}
data {
  int<lower=1> N1;
  int<lower=1> ngroups;
  real x1[N1];
  vector[N1] y1;
  int<lower=1> N2;
  real x2[N2];
  matrix[N1, ngroups] design1;
  matrix[N2, ngroups] design2;
  int groups1[N1];
  int groups2[N2];
  matrix[ngroups, ngroups] group_dist_mat;
}
transformed data {
  real delta = 1e-9;
  int<lower=1> N = N1 + N2;
  vector[N] x;
  int groups[N];
  for (n1 in 1:N1) x[n1] = x1[n1];
  for (n2 in 1:N2) x[N1 + n2] = x2[n2];
  for (n1 in 1:N1) groups[n1] = groups1[n1];
  for (n2 in 1:N2) groups[N1 + n2] = groups2[n2];
  matrix[N, N] squared_dist_mat;
  for (i in 1:N) {
      squared_dist_mat[i, i] = 0;
      for (j in (i + 1):N) {
        squared_dist_mat[i, j] = square(x[i] - x[j]);
        squared_dist_mat[j, i] = squared_dist_mat[i, j];
      }
  }
}
parameters {
  real<lower=0> outputvariance;
  real<lower=0> lengthscale;
  real<lower=0> sigma;
  real<lower=0> alpha;
  vector[ngroups] beta;
  vector[N] eta;
}
transformed parameters {
  cov_matrix[N] K = multigroup_rbf_kernel(squared_dist_mat, groups, group_dist_mat, lengthscale, outputvariance, alpha, delta);
  cholesky_factor_cov[N] L_K = cholesky_decompose(K + sigma * diag_matrix(rep_vector(1.0, N)));
  //vector[N] f = L_K * eta;
}
model {
  outputvariance ~ inv_gamma(5, 5);
  lengthscale ~ inv_gamma(5, 5);
  sigma ~ inv_gamma(1, 1); // make this group-specific
  alpha ~ inv_gamma(5, 5);
  beta ~ std_normal();
  eta ~ std_normal();
  //y1 ~ normal(f[1:N1] + design1 * beta, sigma);
  y1 ~ multi_normal_cholesky(design1 * beta, L_K);
}
generated quantities {
  vector[N1] y1_mean = f[1:N1] + design1 * beta;
  vector[N2] y2;
  for (n2 in 1:N2)
    y2[n2] = normal_rng(f[N1 + n2] + design2[n2,] * beta, sigma);
}



