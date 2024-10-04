library(tidyverse)
library(here)
library(rstan)
library(JQuantsR)
library(logger)

options(mc.cores=parallel::detectCores())
rstan_options(auto_write=TRUE)


JQuantsR::authorize()
df_topix <- JQuantsR::get_indices(code="0000")
df_reit <- JQuantsR::get_indices(code="0075")
df_topix <- df_topix |> 
  mutate(Date=as.Date(Date, "%Y-%m-%d")) |> 
  rename(Close_topix=Close) |> 
  select(Date, Close_topix)
df_reit <- df_reit |> 
  mutate(Date=as.Date(Date, "%Y-%m-%d")) |> 
  rename(Close_reit=Close) |> 
  select(Date, Close_reit)

df <- inner_join(df_topix, df_reit, by="Date") |> 
  mutate(
    ret_topix=(log(Close_topix) - lag(log(Close_topix)))*100,
    ret_reit=(log(Close_reit) - lag(log(Close_reit)))*100
  ) |> 
  slice(-1) %>%
  slice(seq(5, nrow(.), 5))
df

mod <- rstan::stan_model(here("script/correlation/model.stan"))

logger::log_info("start")
fit <- rstan::sampling(
  mod,
  data=list(
    T=nrow(df),
    y=df |> select(ret_reit, ret_topix)
  ),
  chains=4, iter=2000, warmup=1000, thin=1, seed=1234, refresh=10
)
logger::log_info("end")
saveRDS(fit, here("script/correlation/model.rds"))
