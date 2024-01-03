import numpy as np
import scipy as sp
import pandas_datareader.data as pdr
import polars as pl
import patchworklib as pw
from plotnine import *

from kalman_filter import filtering, smoothing, reverse_loglik


# 株価の取得
stock = pdr.DataReader("9501.JP", data_source="stooq", start="2001-01-01", end="2023-12-28")
df_stock = pl.from_pandas(stock.reset_index())
df_stock = (
    df_stock
    .sort("Date")
    # データソース的に数レコードだけ株価がnullの日付があるが、nullの場合は削除する
    .filter(pl.col("Close").is_not_null())
    .with_columns(
        Date=pl.col("Date").dt.date(),
        ret=(pl.col("Close").log() - pl.col("Close").shift(1).log())*100
    )
    .slice(offset=1)
)

# TOPIXなら^TPX
market = pdr.DataReader("^NKX", data_source="stooq", start="2001-01-01", end="2023-12-28")
df_market = pl.from_pandas(market.reset_index())
df_market = (
    df_market
    .sort("Date")
    .filter(pl.col("Close").is_not_null())
    .with_columns(
        Date=pl.col("Date").dt.date(),
        ret=(pl.col("Close").log() - pl.col("Close").shift(1).log())*100
    )
    .slice(offset=1)
)

df = (
    df_stock
    .rename({"Date": "date", "Close": "close_stock", "ret": "ret_stock"})
    .select("date", "close_stock", "ret_stock")
    .join(
        df_market
        .rename({"Date": "date", "Close": "close_market", "ret": "ret_market"})
        .select("date", "close_market", "ret_market"),
        how="inner",
        on="date"
    )
)

# 対数尤度を最大化する観測誤差と状態誤差を得る
ret_market = df.get_column("ret_market").to_numpy()
ret_stock = df.get_column("ret_stock").to_numpy()
y = ret_stock
x = ret_market
T = len(ret_stock)
dims = 2
G = np.eye(dims)
F = np.eye(T, dims)
F[:, 0] = 1
F[:, 1] = x
m0 = np.zeros(dims)
C0 = np.eye(dims)*10000000

best_par=sp.optimize.minimize(
    reverse_loglik,
    [0.0, 0.0],
    args=(dims, y, G, F, m0, C0),
    method="BFGS"
)
W = np.eye(dims) * np.exp(best_par.x[0])
V = np.array([1]).reshape((1, 1)) * np.exp(best_par.x[1])

# 上で求めた観測誤差と状態誤差をもとにフィルタリングと平滑化を行う
m, C = np.zeros((T, dims)), np.zeros((T, dims, dims))
a, R = np.zeros((T, dims)), np.zeros((T, dims, dims))
f, Q = np.zeros((T)), np.zeros((T))
s, S = np.zeros((T, dims)), np.zeros((T, dims, dims))
# フィルタリング
for t in range(0, T):
    _F = F[t].reshape((1, dims))
    if t == 0:
        m[t], C[t], a[t], R[t], f[t], Q[t] = filtering(y[t], m0, C0, G, _F, W, V)
    else:
        m[t], C[t], a[t], R[t], f[t], Q[t] = filtering(y[t], m[t-1], C[t-1], G, _F, W, V)
# 平滑化
for t in range(T - 1, 0, -1):
    if t == T - 1:
        s[t], S[t] = m[t], C[t]
    else:
        s[t], S[t] = smoothing(s[t+1], S[t+1], m[t], C[t], a[t+1], R[t+1], G)

# 推定値と95%信頼区間を取り出す
beta_est = (
    pl.DataFrame({
        "date": df.select("date").get_columns()[0],
        "estimated": m[:, 1],
        "std_error": np.sqrt(C[:, 1, 1])
    })
    .with_columns(
        lower=pl.col("estimated")+sp.stats.norm.ppf(0.025)*pl.col("std_error"),
        upper=pl.col("estimated")+sp.stats.norm.ppf(0.975)*pl.col("std_error"),
    )
)
alpha_est = (
    pl.DataFrame({
        "date": df.select("date").get_columns()[0],
        "estimated": m[:, 0],
        "std_error": np.sqrt(C[:, 0, 0])
    })
    .with_columns(
        lower=pl.col("estimated")+sp.stats.norm.ppf(0.025)*pl.col("std_error"),
        upper=pl.col("estimated")+sp.stats.norm.ppf(0.975)*pl.col("std_error"),
    )
)
beta_smooth = (
    pl.DataFrame({
        "date": df.select("date").get_columns()[0],
        "estimated": s[:, 1],
        "std_error": np.sqrt(S[:, 1, 1])
    })
    .with_columns(
        lower=pl.col("estimated")+sp.stats.norm.ppf(0.025)*pl.col("std_error"),
        upper=pl.col("estimated")+sp.stats.norm.ppf(0.975)*pl.col("std_error"),
    )
)
alpha_smooth = (
    pl.DataFrame({
        "date": df.select("date").get_columns()[0],
        "estimated": s[:, 0],
        "std_error": np.sqrt(S[:, 0, 0])
    })
    .with_columns(
        lower=pl.col("estimated")+sp.stats.norm.ppf(0.025)*pl.col("std_error"),
        upper=pl.col("estimated")+sp.stats.norm.ppf(0.975)*pl.col("std_error"),
    )
)

# 結果のプロット
# 最初の50期はパラメータの推定が安定していないので捨てる
p1 = (
    ggplot(beta_est.slice(50), aes("date"))+
    theme_light()+
    geom_ribbon(aes(ymin="lower", ymax="upper"), fill="lightsteelblue", alpha=0.5)+
    geom_line(aes(y="lower"), color="lightsteelblue")+
    geom_line(aes(y="upper"), color="lightsteelblue")+
    geom_line(aes(y="estimated"), color="firebrick")+
    scale_x_date(breaks="1 year", date_labels="%y")+
    scale_y_continuous(breaks=range(-1, 3, 1))+
    labs(
        title="time-varing beta (filtered); red: estimated, light blue: 95%CI",
        x="date (year)",
        y="beta"
    )
)
p2 = (
    ggplot(beta_smooth.slice(50), aes("date"))+
    theme_light()+
    geom_ribbon(aes(ymin="lower", ymax="upper"), fill="lightsteelblue", alpha=0.5)+
    geom_line(aes(y="lower"), color="lightsteelblue")+
    geom_line(aes(y="upper"), color="lightsteelblue")+
    geom_line(aes(y="estimated"), color="firebrick")+
    scale_x_date(breaks="1 year", date_labels="%y")+
    scale_y_continuous(breaks=range(-1, 3, 1))+
    labs(
        title="time-varing beta (smoothed); red: estimated, light blue: 95%CI",
        x="date (year)",
        y="beta"
    )
)
p3 = (
    ggplot(df.slice(50), aes("date", "close_stock"))+
    theme_light()+
    geom_line()+
    scale_x_date(breaks="1 year", date_labels="%y")+
    labs(
        title="stock price (close)",
        x="date (year)",
        y="close"
    )
)
pw.load_ggplot(p1, figsize=(10, 3)) / pw.load_ggplot(p2, figsize=(10, 3)) / pw.load_ggplot(p3, figsize=(10, 3))
