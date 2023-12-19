data {
  int N;
  vector[N] y;
}

parameters {
  vector[N] x;
  real mu;
  real<lower=-1,upper=1> phi;
  real<lower=0> sigma_eta;
}

transformed parameters {
  real phi_beta;
  phi_beta = (phi + 1) / 2;
  real sigma_eta_square;
  sigma_eta_square = sigma_eta^2;
}

model {
  mu ~ normal(0, 1);
  phi_beta ~ beta(20, 1.5);
  // 5/2は2になるので5.0/2が正しい
  sigma_eta_square ~ inv_gamma(5.0/2, 0.05/2);
  
  // Stanのnormalの引数は分散ではなく標準偏差
  x[1] ~ normal(mu, sigma_eta / sqrt(1 - phi^2));
  x[2:N] ~ normal(mu + phi * (x[1:(N-1)] - mu), sigma_eta);
  y ~ normal(0, exp(x/2));
}

generated quantities {
  vector[N] vol;
  vol = exp(x/2);
}
