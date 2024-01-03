library(tidyverse)
library(quantmod)
library(KFAS)
library(here)


source(here("script/beta-kalman-filter/utils_kfas.R"),encoding="UTF-8")


# データの読み込み ----------------------------------------------------------------
# xts型
stock <- quantmod::getSymbols("9501.T", from="2007-01-01", to="2023-12-29", auto.assign=FALSE)
nikkei <- quantmod::getSymbols("^N225", from="2007-01-01", to="2023-12-29", auto.assign=FALSE)

df_stock <- stock %>% 
  as.data.frame() %>% 
  tibble::rownames_to_column(var="date") %>% 
  purrr::set_names(c("date", "open", "high", "low", "close", "volume", "adjusted")) %>% 
  as_tibble() %>% 
  mutate(date=as.Date(date)) %>% 
  mutate(ret=(log(close) - log(lag(close)))*100)
df_nikkei <- nikkei %>% 
  as.data.frame() %>% 
  tibble::rownames_to_column(var="date") %>% 
  purrr::set_names(c("date", "open", "high", "low", "close", "volume", "adjusted")) %>% 
  as_tibble() %>% 
  mutate(date=as.Date(date)) %>% 
  mutate(ret=(log(close) - log(lag(close)))*100)
  

df <- inner_join(
  df_stock %>% 
    select(date, close, ret) %>% 
    rename(close_stock=close, ret_stock=ret),
  df_nikkei %>% 
    select(date, close, ret) %>% 
    rename(close_market=close, ret_market=ret),
  by="date"
) %>% 
  # 最初の1日目の対数変化率がNAなのを除外
  slice(2:nrow(.))


# KFASでベータの推定 --------------------------------------------------------------------
mod <- KFAS::SSModel(
  # 観測誤差の分散
  H=NA,
  # SSMregression内の-1は状態方程式に切片がないことを、
  # SSMregression外の-1は観測方程式に切片（alpha）がないことを示す
  # Qは状態誤差の分散
  ret_stock ~ KFAS::SSMregression(~ret_market-1, Q=NA),
  data=df
)

# グリッドサーチして尤度が最も大きい初期値を考える
# range_grids <- -2:2
# tictoc::tic()
# このoptimsを見て、尤度が大きい初期値をinits_bestに手で設定するのも良い
# optims <- grid_search_KFAS(mod, range_grids, 2, TRUE)
# tictoc::toc()

# 決め打ち
inits_best <- c(0,0)
fit <- KFAS::fitSSM(mod, inits=inits_best, method="BFGS")
# smoothingにdisturbanceを指定しないとresiduals(res_kfas4,type="state")で状態の誤差を取得できない
kfs <- KFAS::KFS(fit$model, filtering=c("state", "mean"), smoothing=c("state", "mean", "disturbance"))


# 推定されたベータ値を取り出しプロットする --------------------------------------------------------------------
res <- extract_param_kfas(kfs, "ret_market", 0.95) %>% 
  add_column(date=df$date, .before=1)
res <- full_join(df, res, by="date") %>% 
  slice(51:nrow(.))

# geom_ribbon ver
plot_beta <- res %>%
  ggplot(aes(x=date))+
  theme_light()+
  geom_ribbon(aes(ymin=filtered_lower, ymax=filtered_upper), fill="lightsteelblue1", alpha=0.5)+
  geom_line(aes(y=filtered_lower), color="lightsteelblue1", alpha=0.5)+
  geom_line(aes(y=filtered_upper), color="lightsteelblue1", alpha=0.5)+
  geom_line(aes(y=filtered), color="firebrick")+
  scale_x_date(breaks=scales::date_breaks("1 year"), date_labels="%y")+
  scale_y_continuous(breaks=seq(-2, 5, 0.5), minor_breaks=seq(-2, 5, 0.1))+
  labs(x="date", y="beta")
plot_close <- res %>% 
  ggplot(aes(x=date, y=close_stock))+
  theme_light()+
  geom_line()+
  scale_x_date(breaks=scales::date_breaks("1 year"), date_labels="%y")+
  labs(x="date", y="close")
patchwork::wrap_plots(plot_beta, plot_close, ncol=1)
