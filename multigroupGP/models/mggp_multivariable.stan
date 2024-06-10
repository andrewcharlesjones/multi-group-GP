functions {
  matrix multigroup_rbf_kernel(matrix x, int[] groups, matrix group_dist_mat, real lengthscale, real outputvariance, real alpha, real delta) {
    int N = rows(x);
    int P = cols(x);
    matrix[N, N] K;
    for (i in 1:(N - 1)) {
      K[i, i] = 1 + delta;
      for (j in (i + 1):N) {
        int group1 = groups[i] + 1;
        int group2 = groups[j] + 1;
        real group_diff_indicator = group_dist_mat[group1, group2] * 1.0;
        real scaling_term = 1 / pow(group_diff_indicator * square(alpha) + 1, 0.5 * P);
        K[i, j] = scaling_term * exp(-0.5 * sum(square(x[i] - x[j])) * lengthscale / (group_diff_indicator * square(alpha) + 1));
        K[j, i] = K[i, j];
      }
    }
    K[N, N] = 1 + delta;
    return outputvariance * K;
  }
}
data {
  int<lower=1> P;
  int<lower=1> ngroups;
  int<lower=1> N1;
  matrix[N1, P] x1;
  vector[N1] y1;
  int<lower=1> N2;
  matrix[N2, P] x2;
  matrix[N1, ngroups] design1;
  matrix[N2, ngroups] design2;
  int groups1[N1];
  int groups2[N2];
  matrix[ngroups, ngroups] group_dist_mat;
}
transformed data {
  real delta = 1e-8;
  int<lower=1> N = N1 + N2;
  matrix[N, P] x;
  int groups[N];
  for (n1 in 1:N1) x[n1] = x1[n1];
  for (n2 in 1:N2) x[N1 + n2] = x2[n2];
  for (n1 in 1:N1) groups[n1] = groups1[n1];
  for (n2 in 1:N2) groups[N1 + n2] = groups2[n2];
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
  vector[N] f;
  cholesky_factor_cov[N] L_K;
  cov_matrix[N] K = multigroup_rbf_kernel(x, groups, group_dist_mat, lengthscale, outputvariance, alpha, delta);
  L_K = cholesky_decompose(K);
  f = L_K * eta;
}
model {
  outputvariance ~ inv_gamma(5, 5);
  lengthscale ~ inv_gamma(5, 5);
  sigma ~ inv_gamma(1, 1);
  alpha ~ inv_gamma(5, 5);
  beta ~ std_normal();
  eta ~ std_normal();
  y1 ~ normal(f[1:N1] + design1 * beta, sigma);
}
generated quantities {
  vector[N2] y2;
  for (n2 in 1:N2)
    y2[n2] = normal_rng(f[N1 + n2] + design2[n2,] * beta, sigma);
}

