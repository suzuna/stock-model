// 大森（2019）多変量ボラティリティモデルのベイズ推定
// v0: 素直な実装（非中心化なし、境界値配慮なし）
// modelv2 との比較用

data {
  int<lower=0> n; // 時点数
  int<lower=1> p; // 次元（p=2固定）
  matrix[p, n] y; // 収益率データ（2×n行列）
}

parameters {
  // 対数ボラティリティの状態変数（直接サンプリング）
  matrix[p, n] h;
  // 変換前の相関係数の状態変数（直接サンプリング）
  vector[n] g;

  vector[p] mu;
  // phi の raw パラメータ（境界ぴったり）
  vector<lower=0, upper=1>[p] phi_raw;
  // sigma_eta の raw パラメータ
  vector<lower=0>[p] sigma_eta_sq;
  real<lower=0> sigma_zeta;
}

transformed parameters {
  // phi と sigma_eta を raw パラメータから導出
  vector<lower=-1, upper=1>[p] phi = 2 * phi_raw - 1;
  vector<lower=0>[p] sigma_eta = sqrt(sigma_eta_sq);

  // 相関係数
  vector<lower=-1, upper=1>[n] rho;

  // 相関係数の変換
  for (t in 1:n) {
    rho[t] = (exp(g[t]) - 1) / (exp(g[t]) + 1);
  }
}

model {
  // 事前分布（論文準拠、ヤコビアンなし）
  mu ~ normal(0, 1);                      // 論文: N(0, 1)
  phi_raw ~ beta(20, 1.5);                // 論文: (φ+1)/2 ~ Beta(20, 1.5)
  sigma_eta_sq ~ inv_gamma(2.5, 0.025);   // 論文: σ² ~ IG(2.5, 0.025)
  // sigma_zeta: flat prior（何も書かない）

  // 初期値の分布
  for (i in 1:p) {
    h[i, 1] ~ normal(mu[i], sigma_eta[i] / sqrt(1 - phi[i]^2));
  }
  g[1] ~ normal(0, 10);

  // 状態方程式（直接サンプリング）
  for (t in 2:n) {
    for (i in 1:p) {
      h[i, t] ~ normal(mu[i] + phi[i] * (h[i, t-1] - mu[i]), sigma_eta[i]);
    }
    g[t] ~ normal(g[t-1], sigma_zeta);
  }

  // 観測方程式
  for (t in 1:n) {
    real sigma1 = exp(h[1, t] / 2.0);
    real sigma2 = exp(h[2, t] / 2.0);

    // 共分散行列
    matrix[p, p] Sigma;
    Sigma[1, 1] = sigma1^2;
    Sigma[2, 2] = sigma2^2;
    Sigma[1, 2] = rho[t] * sigma1 * sigma2;
    Sigma[2, 1] = Sigma[1, 2];

    y[, t] ~ multi_normal(rep_vector(0.0, p), Sigma);
  }
}

generated quantities {
  // ボラティリティ（パーセント単位）
  matrix[p, n] volatility;

  // 対数尤度
  vector[n] log_lik;

  for (t in 1:n) {
    real sigma1 = exp(h[1, t] / 2.0);
    real sigma2 = exp(h[2, t] / 2.0);

    volatility[1, t] = sigma1;
    volatility[2, t] = sigma2;

    matrix[p, p] Sigma;
    Sigma[1, 1] = sigma1^2;
    Sigma[2, 2] = sigma2^2;
    Sigma[1, 2] = rho[t] * sigma1 * sigma2;
    Sigma[2, 1] = Sigma[1, 2];

    log_lik[t] = multi_normal_lpdf(y[, t] | rep_vector(0.0, p), Sigma);
  }
}
