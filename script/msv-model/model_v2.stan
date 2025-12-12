// 大森（2019）多変量ボラティリティモデルのベイズ推定
// v2: 非中心化パラメータ化 + phi の制約強化

data {
  int<lower=0> n; // 時点数
  int<lower=1> p; // 次元（p=2固定）
  matrix[p, n] y; // 収益率データ（2×n行列）
}

parameters {
  // 非中心化パラメータ（標準正規分布からサンプル）
  matrix[p, n] h_raw;
  vector[n] g_raw;

  vector[p] mu;
  // phi の raw パラメータ（phi_raw = (phi+1)/2）
  vector<lower=0.0005, upper=0.9995>[p] phi_raw;
  // sigma_eta の raw パラメータ（sigma_eta_sq = sigma_eta^2）
  vector<lower=0>[p] sigma_eta_sq;
  real<lower=0> sigma_zeta;
}

transformed parameters {
  // phi と sigma_eta を raw パラメータから導出（h の計算より前に！）
  vector<lower=-1, upper=1>[p] phi = 2 * phi_raw - 1;
  vector<lower=0>[p] sigma_eta = sqrt(sigma_eta_sq);

  // 対数ボラティリティの状態変数（非中心化から変換）
  matrix[p, n] h;
  // 変換前の相関係数の状態変数（非中心化から変換）
  vector[n] g;
  // 相関係数
  vector<lower=-1, upper=1>[n] rho;

  // 非中心化パラメータ化
  // h の初期値
  for (i in 1:p) {
    h[i, 1] = mu[i] + (sigma_eta[i] / sqrt(1 - phi[i]^2)) * h_raw[i, 1];
  }
  // g の初期値
  g[1] = 10 * g_raw[1];  // 元の事前分布 normal(0, 10) に対応

  // 状態方程式（非中心化）
  for (t in 2:n) {
    for (i in 1:p) {
      h[i, t] = mu[i] + phi[i] * (h[i, t-1] - mu[i]) + sigma_eta[i] * h_raw[i, t];
    }
    g[t] = g[t-1] + sigma_zeta * g_raw[t];
  }

  // 相関係数の変換（Fisher's Z変換）
  for (t in 1:n) {
    rho[t] = tanh(g[t] / 2);  // より安定な計算
  }
}

model {
  // 事前分布（論文準拠）
  mu ~ normal(0, 1);                      // 論文: N(0, 1)
  phi_raw ~ beta(20, 1.5);                // 論文: (φ+1)/2 ~ Beta(20, 1.5)
  sigma_eta_sq ~ inv_gamma(2.5, 0.025);   // 論文: σ² ~ IG(2.5, 0.025)
  // sigma_zeta: flat prior（何も書かない）

  // 非中心化パラメータは標準正規分布
  to_vector(h_raw) ~ std_normal();
  g_raw ~ std_normal();

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
