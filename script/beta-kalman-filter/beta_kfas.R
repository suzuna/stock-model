library(tidyverse)
library(lubridate)
library(KFAS)
library(here)


source(here("script/utils.R"),encoding="UTF-8")
source(here("script/utils_kfas.R"),encoding="UTF-8")


# データの読み込み ----------------------------------------------------------------
tepco <- read_stockcsv_daily(here("data/9501_tepcoHD.csv"),"9501_tepcoHD")
tokyogas <- read_stockcsv_daily(here("data/9531_tokyogas.csv"),"9531_tokyogas")
topix <- read_stockcsv_daily(here("data/topix.csv"),"0000_topix")
nikkei <- read_stockcsv_daily(here("data/nikkei.csv"),"0001_nikkei")

df <- left_join(
  tepco %>% 
    select(date,close,ret),
  topix %>% 
    select(date,close,ret) %>% 
    rename(close_topix=close,ret_topix=ret),
  by="date"
) %>% 
  # 最初の1日目の対数変化率がNAなのを除外
  slice(2:nrow(.))


# KFASでベータの推定 --------------------------------------------------------------------
mod <- SSModel(
  H=NA,
  ret ~ SSMregression(~ret_topix-1,Q=NA)-1,
  data=df
)

# グリッドサーチして尤度が最も大きい初期値を考える
# range_grids <- -2:2
# tictoc::tic()
# このoptimsを見て、尤度が大きい初期値をinits_bestに手で設定するのも良い
# optims <- grid_search_KFAS(mod,range_grids,2,TRUE)
# tictoc::toc()

# 決め打ち
inits_best <- c(0,0)
fit <- fitSSM(mod,inits=inits_best,method="BFGS")
# smoothingにdisturbanceを指定しないとresiduals(res_kfas4,type="state")で状態の誤差を取得できない
# kfs <- KFS(fit$model,filtering=c("state","mean"),smoothing=c("state","mean","disturbance"))
kfs <- KFS(fit$model,filtering=c("state","mean"),smoothing=c("state","mean"))
kfs


# 推定されたベータ値を取り出しプロットする --------------------------------------------------------------------
res <- extract_param_kfas(kfs,"ret_topix",0.95) %>% 
  add_column(date=df$date,.before=1)
res <- full_join(df,res,by="date") %>% 
  slice(51:nrow(.))

# res %>%
#   slice(50:nrow(.)) %>%
#   select(date,filtered,filtered_upper,filtered_lower) %>%
#   pivot_longer(cols=-date,names_to="param",values_to="value") %>%
#   ggplot(aes(date,value,color=param))+geom_line()+theme_light()+
#   scale_x_date(breaks=scales::date_breaks("1 year"),date_labels="%y")+
#   ggsci::scale_color_aaas()

# geom_ribbon ver
plot_beta <- res %>%
  select(date,filtered,filtered_upper,filtered_lower) %>%
  ggplot(aes(x=date))+theme_light()+
  geom_ribbon(aes(ymin=filtered_lower,ymax=filtered_upper),fill="lightsteelblue1",alpha=0.5)+
  geom_line(aes(y=filtered_lower),color="lightsteelblue1",alpha=0.5)+
  geom_line(aes(y=filtered_upper),color="lightsteelblue1",alpha=0.5)+
  geom_line(aes(y=filtered),color="firebrick")+
  scale_x_date(breaks=scales::date_breaks("1 year"),date_labels="%y")+
  scale_y_continuous(breaks=seq(-2,5,0.5),minor_breaks=seq(-2,5,0.1))+
  labs(x="date",y="beta")
  # geom_vline(xintercept=as.Date("2011-03-11"))
  # annotate("label",x=df2$date[1],y=Inf,label="red: filtered (estimated)\nblue: ±95%CI",hjust=0,vjust=1.5,alpha=0)

plot_close <- res %>% 
  select(date,close) %>% 
  ggplot(aes(x=date,y=close))+theme_light()+geom_line()+
  scale_x_date(breaks=scales::date_breaks("1 year"),date_labels="%y")+
  labs(x="date",y="close")
patchwork::wrap_plots(plot_beta,plot_close,ncol=1)
