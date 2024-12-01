import datetime
import json
import time

import polars as pl
import requests
from tqdm import tqdm


endpoint = "https://forex-api.coin.z.com/public/v1/klines"
dates = [i.strftime("%Y%m%d") for i in pl.date_range(
    start=datetime.date(2023, 10, 28),
    end=datetime.date(2024, 12, 1),
    interval="1d",
    eager=True
)]
dates

res = []
for date in tqdm(dates):
    params = {
        "symbol": "USD_JPY",
        "priceType": "ASK",
        "interval": "5min",
        "date": date,
    }
    resp = requests.get(endpoint, params=params)
    # データが存在しない日（市場が開いていない日）は空のリスト("[]")のままappendする
    res.append(pl.DataFrame(json.loads(resp.text)["data"]))
    # time.sleep(1)
df = pl.concat([i for i in res if not i.is_empty()])
df

df = (
    df
    .with_columns(
        timestamp=pl.from_epoch(pl.col("openTime").cast(int), time_unit="ms").dt.strftime("%Y-%m-%dT%H:%M:%S+09:00"),
        open=pl.col("open").cast(float),
        high=pl.col("high").cast(float),
        low=pl.col("low").cast(float),
        close=pl.col("close").cast(float),
    )
)
df.write_csv("usdjpy_5min_20231029_20241201.csv")
