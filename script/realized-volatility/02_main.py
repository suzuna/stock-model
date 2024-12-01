import datetime
import math

import patchworklib as pw
import plotnine as pn
import polars as pl
import scipy as sp


df = pl.read_csv("../../data/usdjpy_5min_20231029_20241204.csv")

# 元のtimestamp列はUTCっぽい
# JSTの6:00がその日の始まり（月曜日は7:00）
df = (
    df
    .rename({"openTime": "openTimeUtc"})
    .with_columns(
        timestamp_utc=pl.from_epoch(pl.col("openTimeUtc").cast(int), time_unit="ms")
    )
    .with_columns(
        timestamp_jst=pl.col("timestamp_utc")+datetime.timedelta(hours=9),
        timestamp=pl.col("timestamp_utc")+datetime.timedelta(hours=9)-datetime.timedelta(hours=6)
    )
    .with_columns(date=pl.col("timestamp").dt.date())
)
df

mu_1 = 2**(1/2) * math.gamma(1) * math.gamma(1/2)**(-1)
mu_4over3 = 2**(2/3) * math.gamma(7/6) * math.gamma(1/2)**(-1)
alpha = 0.95

df_volatility = (
    df
    .with_columns(ret=(pl.col("close").log() - pl.col("close").shift(1).log()) * 100)
    .group_by("date")
    .agg(
        n=pl.len(),
        rv=(pl.col("ret")**2).sum(),
        bv=mu_1**(-2) * (pl.col("ret").abs() * pl.col("ret").shift(1).abs()).sum(),
        tq=pl.len() * mu_4over3**(-3) * (pl.col("ret").abs()**(4/3) * pl.col("ret").shift(1).abs()**(4/3) * pl.col("ret").shift(2).abs()**(4/3)).sum(),
    )
    .sort("date")
    .with_columns(
        z=(pl.col("rv").log() - pl.col("bv").log()) / ((mu_1**(-4) + 2 * mu_1**(-2) - 5) * pl.col("tq") * pl.col("bv")**(-2) / pl.col("n"))**(1/2)
    )
    .with_columns(
        j=pl.when(pl.col("z") > sp.stats.norm.ppf(alpha)).then(pl.col("rv") - pl.col("bv")).otherwise(pl.lit(0))
    )
    .with_columns(
        c=pl.col("rv") - pl.col("j")
    )
)

p1 = (
    pn.ggplot(
        df_volatility.select("date", "j", "c")
        .melt(id_vars="date", variable_name="variable", value_name="value"),
    )
    +pn.geom_area(pn.aes("date", "value", fill="variable"), color="gray", size=0.2, alpha=0.8)
    +pn.scale_fill_brewer(type="qual", palette="Set2")
    +pn.scale_x_date(date_breaks="3 month", date_minor_breaks="1 month", date_labels="%Y/%m/%d")
    +pn.scale_y_continuous(breaks=range(0, 100, 1))
    +pn.theme_minimal()
    +pn.labs(x="date", y="volatility")
)
p1
p2 = (
    pn.ggplot(df)
    +pn.geom_line(pn.aes("timestamp", "close"))
    +pn.scale_x_date(date_breaks="3 month", date_minor_breaks="1 month", date_labels="%Y/%m/%d")
    +pn.theme_minimal()
)
p2

# 4/29と7/11は介入の日
# https://www.asahi.com/articles/ASSC80H31SC8ULFA003M.html
(
    pn.ggplot(df.filter(pl.col("date") == datetime.date(2024, 4, 29)))
    +pn.geom_line(pn.aes("timestamp_jst", "close"))
    +pn.scale_x_datetime(date_breaks="6 hours", date_minor_breaks="1 hours", date_labels="%m/%d %H:%M")
)

(
    df_volatility
    .select(
        days_jump=(pl.col("j") > 0).sum(),
        days_no_jump=(pl.col("j") == 0).sum()
    )
)

# ----------
# 非営業日を外すテスト
dates = list(set(df.filter(pl.col("date") <= datetime.date(2023, 11, 15)).get_column("date")))
dates
# list(dates)
p2 = (
    pn.ggplot(df.filter(pl.col("timestamp") <= datetime.date(2023, 11, 15)))
    +pn.geom_line(pn.aes("timestamp", "close"))
    # +pn.scale_x_date(breaks=dates, labels=dates)
)
p2
