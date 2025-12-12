// 大森（2019）多変量ボラティリティモデルのベイズ推定
// v4: 非中心化パラメータ化 + レバレッジ効果 + multi_normal_cholesky
// 参考: 大森・渡部（2007）CARF-J-035

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

  // レバレッジパラメータ（ε_t と η_t の相関）→ flat prior
  vector<lower=-0.999, upper=0.999>[p] rho_leverage;
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
  // 標準化観測誤差
  matrix[p, n] epsilon;

  // h の初期値（定常分布から）
  for (i in 1:p) {
    h[i, 1] = mu[i] + (sigma_eta[i] / sqrt(1 - phi[i]^2)) * h_raw[i, 1];
  }
  // g の初期値
  g[1] = 10 * g_raw[1];  // 元の事前分布 normal(0, 10) に対応

  // 初期時点の標準化観測誤差（fmaxでゼロ除算防止）
  for (i in 1:p) {
    real vol = fmax(exp(h[i, 1] / 2.0), 1e-10);
    epsilon[i, 1] = y[i, 1] / vol;
  }

  // 状態方程式（非中心化 + レバレッジ効果）
  for (t in 2:n) {
    for (i in 1:p) {
      // レバレッジ項: ε_{t-1} に依存する部分
      real leverage_term = rho_leverage[i] * sigma_eta[i] * epsilon[i, t-1];
      // 独立項: ε_{t-1} と独立な部分
      real independent_term = sqrt(1 - rho_leverage[i]^2) * sigma_eta[i] * h_raw[i, t];
      // 状態方程式
      h[i, t] = mu[i] + phi[i] * (h[i, t-1] - mu[i]) + leverage_term + independent_term;
      // 標準化観測誤差を計算（次の時点で使う、fmaxでゼロ除算防止）
      real vol_t = fmax(exp(h[i, t] / 2.0), 1e-10);
      epsilon[i, t] = y[i, t] / vol_t;
    }
    g[t] = g[t-1] + sigma_zeta * g_raw[t];
  }

  // 相関係数の変換（Fisher's Z変換）
  for (t in 1:n) {
    rho[t] = tanh(g[t] / 2);
  }
}

model {
  // 事前分布（論文準拠）
  mu ~ normal(0, 1);                      // 論文: N(0, 1)
  phi_raw ~ beta(20, 1.5);                // 論文: (φ+1)/2 ~ Beta(20, 1.5)
  sigma_eta_sq ~ inv_gamma(2.5, 0.025);   // 論文: σ² ~ IG(2.5, 0.025)
  // sigma_zeta: flat prior（何も書かない）
  // rho_leverage: flat prior（論文: U(-1, 1)）

  // 非中心化パラメータは標準正規分布
  to_vector(h_raw) ~ std_normal();
  g_raw ~ std_normal();

  // 観測方程式（Cholesky分解版）
  for (t in 1:n) {
    real sigma1 = exp(h[1, t] / 2.0);
    real sigma2 = exp(h[2, t] / 2.0);

    // Cholesky因子を直接構築（数値安定性向上）
    matrix[p, p] L_Sigma;
    L_Sigma[1, 1] = sigma1;
    L_Sigma[2, 1] = rho[t] * sigma2;
    L_Sigma[1, 2] = 0;
    L_Sigma[2, 2] = sigma2 * sqrt(1 - rho[t]^2);

    y[, t] ~ multi_normal_cholesky(rep_vector(0.0, p), L_Sigma);
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

    // Cholesky因子を直接構築
    matrix[p, p] L_Sigma;
    L_Sigma[1, 1] = sigma1;
    L_Sigma[2, 1] = rho[t] * sigma2;
    L_Sigma[1, 2] = 0;
    L_Sigma[2, 2] = sigma2 * sqrt(1 - rho[t]^2);

    log_lik[t] = multi_normal_cholesky_lpdf(y[, t] | rep_vector(0.0, p), L_Sigma);
  }
}
