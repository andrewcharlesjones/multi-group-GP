functions {
  matrix rbf_kernel(vector x, real length_scale, real output_variance, real delta) {
    int N = rows(x);
    matrix[N, N] K;
    for (i in 1:(N - 1)) {
      K[i, i] = 1 + delta;
      for (j in (i + 1):N) {
        K[i, j] = exp(-0.5 * square(x[i] - x[j]) * length_scale);
        K[j, i] = K[i, j];
      }
    }
    K[N, N] = 1 + delta;
    return output_variance * K;
  }
  matrix multigroup_rbf_kernel(vector x, vector groups, real length_scale, real output_variance, real alpha, real delta) {
    int N = rows(x);
    matrix[N, N] K;
    for (i in 1:(N - 1)) {
      K[i, i] = 1 / pow(square(groups[i] - groups[i]) * square(alpha) + 1, 0.5) + delta;
      for (j in (i + 1):N) {
        real group_diff_indicator = square(groups[i] - groups[j]);
        real scaling_term = 1 / pow(group_diff_indicator * square(alpha) + 1, 0.5);
        K[i, j] = scaling_term * exp(-0.5 * square(x[i] - x[j]) * length_scale / (group_diff_indicator * square(alpha) + 1));
        K[j, i] = K[i, j];
      }
    }
    K[N, N] = 1 / pow(square(groups[N] - groups[N]) * square(alpha) + 1, 0.5) + delta;
    return output_variance * K;
  }
}
data {
  int<lower=1> N1;
  real x1[N1];
  vector[N1] y1;
  int<lower=1> N2;
  real x2[N2];
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
  real<lower=0> output_variance;
  real<lower=0> length_scale;
  real<lower=0> sigma;
  real<lower=0> alpha;
  vector[N] eta;
}
transformed parameters {
  vector[N] f;
  {
    matrix[N, N] L_K;
    matrix[N, N] K = multigroup_rbf_kernel(x, groups, length_scale, output_variance, alpha, delta);
    L_K = cholesky_decompose(K);
    f = L_K * eta;
  }
}
model {
  output_variance ~ inv_gamma(5, 5);
  length_scale ~ std_normal();
  sigma ~ std_normal();
  alpha ~ inv_gamma(5, 5);
  eta ~ std_normal();
  y1 ~ normal(f[1:N1], sigma);
}
generated quantities {
  vector[N2] y2;
  for (n2 in 1:N2)
    y2[n2] = normal_rng(f[N1 + n2], sigma);
}

