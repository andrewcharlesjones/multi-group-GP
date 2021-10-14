functions {
  matrix rbf_kernel(vector x, real lengthscale, real outputvariance, real delta) {
    int N = rows(x);
    matrix[N, N] K;
    for (i in 1:(N - 1)) {
      K[i, i] = 1 + delta;
      for (j in (i + 1):N) {
        K[i, j] = exp(-0.5 * square(x[i] - x[j]) * lengthscale);
        K[j, i] = K[i, j];
      }
    }
    K[N, N] = 1 + delta;
    return outputvariance * K;
  }
  matrix multigroup_rbf_kernel(vector x, vector groups, real lengthscale, real outputvariance, real alpha, real delta) {
    int N = rows(x);
    matrix[N, N] K;
    for (i in 1:(N - 1)) {
      K[i, i] = 1 / pow(square(groups[i] - groups[i]) * square(alpha) + 1, 0.5) + delta;
      for (j in (i + 1):N) {
        real group_diff_indicator = square(groups[i] - groups[j]);
        real scaling_term = 1 / pow(group_diff_indicator * square(alpha) + 1, 0.5);
        K[i, j] = scaling_term * exp(-0.5 * square(x[i] - x[j]) * lengthscale / (group_diff_indicator * square(alpha) + 1));
        K[j, i] = K[i, j];
      }
    }
    K[N, N] = 1 / pow(square(groups[N] - groups[N]) * square(alpha) + 1, 0.5) + delta;
    return outputvariance * K;
  }
}
data {
  int<lower=1> N1;
  real x1[N1];
  vector[N1] y1;
  int<lower=1> N2;
  real x2[N2];
  matrix[N1, 2] design1;
  matrix[N2, 2] design2;
  vector[N1] groups1;
  vector[N2] groups2;
}
transformed data {
  real delta = 1e-9;
  int<lower=1> N = N1 + N2;
  vector[N] x;
  vector[N] groups;
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
  vector[2] beta;
  vector[N] eta;
}
transformed parameters {
  vector[N] f;
  {
    matrix[N, N] L_K;
    matrix[N, N] K = multigroup_rbf_kernel(x, groups, lengthscale, outputvariance, alpha, delta);
    L_K = cholesky_decompose(K);
    f = L_K * eta;
  }
}
model {
  outputvariance ~ inv_gamma(5, 5);
  lengthscale ~ inv_gamma(1, 1);
  sigma ~ inv_gamma(1, 1);
  alpha ~ inv_gamma(5, 5);
  beta[1] ~ std_normal();
  beta[2] ~ std_normal();
  eta ~ std_normal();
  y1 ~ normal(f[1:N1] + design1 * beta, sigma);
}
generated quantities {
  vector[N2] y2;
  for (n2 in 1:N2)
    y2[n2] = normal_rng(f[N1 + n2] + design2[n2,] * beta, sigma);
}

