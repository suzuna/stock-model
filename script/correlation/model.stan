data {
  int<lower=0> T;
  vector[2] y[T];
}

parameters {
  real<lower=0> sigma1;
  real<lower=0> sigma2;
  real<lower=-1, upper=1> rho[T];
  real<lower=0> rho_sigma;
}

transformed parameters {
  cov_matrix[2] cov[T];
  vector[2] mu;

  for (t in 1:T) {
    cov[t][1, 1] = sigma1^2;
    cov[t][1, 2] = sigma1*sigma2*rho[t];
    cov[t][2, 1] = sigma1*sigma2*rho[t];
    cov[t][2, 2] = sigma2^2;
  }
  mu[1] = 0;
  mu[2] = 0;
}

model {
  for (t in 2:T) {
    rho[t] ~ normal(rho[t-1], rho_sigma);
  }
  for (t in 1:T) {
    y[t] ~ multi_normal(mu, cov[t]);
  }
}

generated quantities {
  vector[2] y_pred[T];
  for (t in 1:T) {
    y_pred[t] = multi_normal_rng(mu, cov[t]);
  }
}
