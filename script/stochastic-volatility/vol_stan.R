library(tidyverse)
library(rstan)
library(bayesplot)
library(here)
library(logger)


# Stanのおまじない（上は並列化、下はstanコードが変わらない限り再コンパイルしない）
# options(mc.cores=parallel::detectCores())
options(mc.cores=4)
rstan_options(auto_write=TRUE)


# データ取得 -------------------------------------------------------------------
# J-Quantsに登録したメールアドレスとパスワード
mail_address <- "MAIL_ADDRESS"
password <- "PASSWORD"

resp <- httr::POST(
  "https://api.jquants.com/v1/token/auth_user",
  body=jsonlite::toJSON(
    list(mailaddress=mail_address, password=password),
    auto_unbox=TRUE
  )
)
refresh_token <- httr::content(resp)$refreshToken

resp <- httr::POST(
  "https://api.jquants.com/v1/token/auth_refresh",
  query=list(refreshtoken=refresh_token)
)
id_token <- httr::content(resp)$idToken

resp <- httr::GET(
  "https://api.jquants.com/v1/indices/topix",
  query=list(from="2008-05-07", to="2023-12-18"),
  httr::add_headers(Authorization=glue::glue("Bearer {id_token}"))
)
topix <- httr::content(resp)$topix |> 
  dplyr::bind_rows() |> 
  tibble::as_tibble()

df <- topix |> 
  mutate(Date=as.Date(Date, "%Y-%m-%d")) |> 
  mutate(ret=(log(Close) - log(lag(Close, 1)))*100) |> 
  slice(-1)


# Stanでの推定 ----------------------------------------------------------------
mod <- rstan::stan_model(here("script/stochastic-volatility/svmodel.stan"))

logger::log_info("start")
fit <- rstan::sampling(
  mod,
  data=list(N=nrow(df), y=df$ret),
  chains=4, iter=11000, warmup=1000, thin=1, seed=1234
)
logger::log_info("end")
saveRDS(fit, here("sv_model.rds"))


# モデル診断 -------------------------------------------------------------------
# fit <- readRDS(here("sv_model.rds"))
bayesplot::mcmc_rhat_hist(bayesplot::rhat(fit))
bayesplot::mcmc_neff_hist(bayesplot::neff_ratio(fit))
bayesplot::mcmc_acf_bar(fit, pars=c("mu", "phi", "sigma_eta"))
bayesplot::mcmc_trace(fit, pars=c("mu", "phi", "sigma_eta"))


# パラメータのプロット --------------------------------------------------------------
mat <- rstan::extract(fit, "vol")[[1]]
vol_stat <- tibble::tibble(
  vol_median=apply(mat, 2, \(x) quantile(x, 0.5)),
  vol_mean=apply(mat, 2, \(x) mean(x)),
  vol_lower=apply(mat, 2, \(x) quantile(x, 0.025)),
  vol_upper=apply(mat, 2, \(x) quantile(x, 0.975))
) |> 
  mutate(Date=df$Date) |> 
  relocate(Date)
res <- left_join(df, vol_stat, by="Date")

p_vol <- res |> 
  ggplot(aes(Date))+
  theme_light()+
  geom_ribbon(aes(ymin=vol_lower, ymax=vol_upper), fill="lightsteelblue1", alpha=0.5)+
  geom_line(aes(y=vol_upper), color="lightsteelblue1")+
  geom_line(aes(y=vol_lower), color="lightsteelblue1")+
  geom_line(aes(y=vol_median), color="firebrick")+
  scale_x_date(breaks=scales::date_breaks("1 year"), date_labels="%y")+
  labs(
    x="date (year)",
    y="volatility (sigma_t)",
    subtitle="red: estimated (median), light blue: 95% CI"
  )
p_topix <- res |> 
  ggplot(aes(x=Date, y=Close))+
  theme_light()+
  geom_line()+
  scale_x_date(breaks=scales::date_breaks("1 year"), date_labels="%y")+
  labs(x="date (year)", y="TOPIX close")
patchwork::wrap_plots(p_vol, p_topix, ncol=1)

print(fit, pars="phi", digits=3)
bayesplot::mcmc_hist(fit, pars="phi")
