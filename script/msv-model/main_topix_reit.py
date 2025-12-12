import datetime
import os

import arviz as az
import cmdstanpy
import numpy as np
import polars as pl
import plotnine as p9
import logging

import jquantsapi

# https://www.carf.e.u-tokyo.ac.jp/old/pdf/workingpaper/jseries/35.pdf
# https://www.boj.or.jp/research/wps_rev/rev_2013/data/rev13j08.pdf
# https://www.carf.e.u-tokyo.ac.jp/old/pdf/workingpaper/jseries/J104.pdf

# loggerの定義
logger = logging.getLogger("cmdstanpy")
logger.disabled = False
logger.handlers = []
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler("cmdstanpy_debug.log")
stream = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', "%H:%M:%S"))
stream.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', "%H:%M:%S"))
logger.addHandler(handler)
logger.addHandler(stream)

# データの取得
cli = jquantsapi.Client(mail_address=os.environ["JQUANTS_EMAIL"], password=os.environ["JQUANTS_PASSWORD"])
df_topix = (
    pl.from_pandas(cli.get_indices(code="0000"))
    .with_columns(Date=pl.col("Date").cast(pl.Date))
)
df_reit = (
    pl.from_pandas(cli.get_indices(code="0075"))
    .with_columns(Date=pl.col("Date").cast(pl.Date))
)
df = (
    df_topix
    .rename({"Close": "CloseTopix"})
    .with_columns(RetTopix=(pl.col("CloseTopix").log() - pl.col("CloseTopix").log().shift(1))*100)
    .select("Date", "CloseTopix", "RetTopix")
    .join(
        df_reit
        .rename({"Close": "CloseReit"})
        .with_columns(RetReit=(pl.col("CloseReit").log() - pl.col("CloseReit").log().shift(1))*100)
        .select("Date", "CloseReit", "RetReit"),
        on="Date", how="inner"
    )
    .sort("Date")
    .slice(offset=1)
    .filter((
        (pl.col("Date") >= datetime.date(2008, 5, 8)) &
        (pl.col("Date") <= datetime.date(2025, 12, 5))
    ))
)

# stanの実行
y_data = df.select(["RetTopix", "RetReit"]).to_numpy().T
n = y_data.shape[1]
data = {"n": n, "p": 2, "y": y_data}

model = cmdstanpy.CmdStanModel(stan_file="model_v0.stan")
fit = model.sample(
    data=data,
    chains=4,
    parallel_chains=4,
    iter_warmup=1000,
    iter_sampling=1000,
    thin=1,
    seed=1234,
    refresh=10,
    show_console=True,
    show_progress=True,
)
logger.info(fit.time)
idata = az.from_cmdstanpy(fit)
idata.to_netcdf("fit_arviz_1_v0.nc")
# idata = az.from_netcdf("fit_arviz_1_v0.nc")

model = cmdstanpy.CmdStanModel(stan_file="model_v1.stan")
fit = model.sample(
    data=data,
    chains=4,
    parallel_chains=4,
    iter_warmup=1000,
    iter_sampling=1000,
    thin=1,
    seed=1234,
    refresh=10,
    show_console=True,
    show_progress=True,
)
logger.info(fit.time)
idata = az.from_cmdstanpy(fit)
idata.to_netcdf("fit_arviz_1_v1.nc")
# idata = az.from_netcdf("fit_arviz_1_v1.nc")

model = cmdstanpy.CmdStanModel(stan_file="model_v2.stan")
fit = model.sample(
    data=data,
    chains=4,
    parallel_chains=4,
    iter_warmup=1000,
    iter_sampling=1000,
    thin=1,
    seed=1234,
    refresh=10,
    show_console=True,
    show_progress=True,
)
logger.info(fit.time)
idata = az.from_cmdstanpy(fit)
idata.to_netcdf("fit_arviz_1_v2.nc")
# idata = az.from_netcdf("fit_arviz_1_v2.nc")

model = cmdstanpy.CmdStanModel(stan_file="model_v2_1.stan")
fit = model.sample(
    data=data,
    chains=4,
    parallel_chains=4,
    iter_warmup=1000,
    iter_sampling=1000,
    thin=1,
    seed=1234,
    refresh=10,
    show_console=True,
    show_progress=True,
)
logger.info(fit.time)
idata = az.from_cmdstanpy(fit)
idata.to_netcdf("fit_arviz_1_v2_1.nc")
# idata = az.from_netcdf("fit_arviz_1_v2_1.nc")

model = cmdstanpy.CmdStanModel(stan_file="model_v3.stan")
fit = model.sample(
    data=data,
    chains=4,
    parallel_chains=4,
    iter_warmup=1000,
    iter_sampling=1000,
    thin=1,
    seed=1234,
    refresh=10,
    show_console=True,
    show_progress=True,
)
logger.info(fit.time)
idata = az.from_cmdstanpy(fit)
idata.to_netcdf("fit_arviz_1_v3.nc")
# idata = az.from_netcdf("fit_arviz_1_v3.nc")

model = cmdstanpy.CmdStanModel(stan_file="model_v4.stan")
fit = model.sample(
    data=data,
    chains=4,
    parallel_chains=4,
    iter_warmup=1000,
    iter_sampling=1000,
    inits=0.1,
    thin=1,
    seed=1234,
    refresh=10,
    show_console=True,
    show_progress=True,
)
logger.info(fit.time)
idata = az.from_cmdstanpy(fit)
idata.to_netcdf("fit_arviz_1_v4.nc")
# idata = az.from_netcdf("fit_arviz_1_v4.nc")

# 事後診断
params_to_plot = ["mu", "phi", "sigma_eta", "sigma_zeta"]
az.plot_trace(idata, var_names=params_to_plot, figsize=(14, 12))
az.plot_posterior(idata, var_names=params_to_plot, figsize=(14, 10))
az.summary(idata, var_names=params_to_plot)

# 結果のプロット
vol = idata.posterior["volatility"].stack(sample=("chain", "draw"))
rho = idata.posterior["rho"].stack(sample=("chain", "draw"))

vol_topix = vol.isel(volatility_dim_0=0).values.T  # (n, sample) -> (sample, n)
vol_reit = vol.isel(volatility_dim_0=1).values.T
rho_samples = rho.values.T

res = (
    pl.DataFrame({
        "Date": df["Date"].to_list(),
        "VolatilityTopixMedian": np.percentile(vol_topix, 50.0, axis=0),
        "VolatilityTopixLower": np.percentile(vol_topix, 2.5, axis=0),
        "VolatilityTopixUpper": np.percentile(vol_topix, 97.5, axis=0),
        "VolatilityReitMedian": np.percentile(vol_reit, 50.0, axis=0),
        "VolatilityReitLower": np.percentile(vol_reit, 2.5, axis=0),
        "VolatilityReitUpper": np.percentile(vol_reit, 97.5, axis=0),
        "RhoMedian": np.percentile(rho_samples, 50.0, axis=0),
        "RhoLower": np.percentile(rho_samples, 2.5, axis=0),
        "RhoUpper": np.percentile(rho_samples, 97.5, axis=0),
    })
)
res_rolling = (
    df
    .with_columns(
        RhoRolling=pl.rolling_corr(pl.col("RetTopix"), pl.col("RetReit"), window_size=250)
    )
    .select(["Date", "RhoRolling"])
)
res_joined = (
    res
    .join(res_rolling, on="Date", how="left")
)
res_joined

(
    p9.ggplot(res)
    + p9.geom_ribbon(
        p9.aes(x="Date", ymin="VolatilityTopixLower", ymax="VolatilityTopixUpper"),
        fill="blue", alpha=0.2
    )
    + p9.geom_line(
        p9.aes(x="Date", y="VolatilityTopixMedian"),
        color="blue"
    )
    + p9.geom_ribbon(
        p9.aes(x="Date", ymin="VolatilityReitLower", ymax="VolatilityReitUpper"),
        fill="red", alpha=0.2
    )
    + p9.geom_line(
        p9.aes(x="Date", y="VolatilityReitMedian"),
        color="red"
    )
    + p9.labs(
        title="Estimated Volatilities",
        y="Volatility (%)"
    )
)


(
    p9.ggplot(res_joined)
    + p9.theme_light()
    + p9.geom_ribbon(
        p9.aes(x="Date", ymin="RhoLower", ymax="RhoUpper"),
        fill="green", alpha=0.2
    )
    + p9.geom_line(
        p9.aes(x="Date", y="RhoMedian"),
        color="green"
    )
    # + p9.geom_line(
    #     p9.aes(x="Date", y="RhoRolling"),
    #     color="red"
    # )
    + p9.scale_x_date(date_labels="%Y/%m", date_breaks="3 year", date_minor_breaks="1 year")
    + p9.scale_y_continuous(breaks=np.arange(-0.5, 1.1, 0.5).tolist(), minor_breaks=np.arange(-0.5, 1.1, 0.1).tolist())
    + p9.labs(y="rho")
)
