library(tidyverse)
library(lubridate)
library(KFAS)
library(here)
library(rstan)
options(mc.cores=parallel::detectCores())
rstan_options(auto_write=TRUE)


source(here("script/utils.R"),encoding="UTF-8")
source(here("script/utils_kfas.R"),encoding="UTF-8")

# http://www.computer-services.e.u-tokyo.ac.jp/p/cemano/research/DP/documents/coe-j-46.pdf
# http://apps.olin.wustl.edu/faculty/chib/papers/KimShephardChib98.pdf


# データの読み込み ----------------------------------------------------------------
topix <- read_stockcsv_daily(here("data/topix.csv"),"0000_topix")
df <- topix %>% 
  slice(2:nrow(.))


# 推定 ----------------------------------------------------------------------
mod <- rstan::stan(
  here("script/svmodel.stan"),
  data=list(
    N=nrow(df),
    y=df$ret
  ),
  chains=4,
  iter=30000,
  warmup=15000,
  seed=1234
)


# 結果を取り出す -----------------------------------------------------------------
extract_param_stan <- function(model,param_name,confidence_interval=0.95) {
  mat <- rstan::extract(model,param_name)[[1]]
  
  ci_upper <- confidence_interval+(1-confidence_interval)/2
  ci_lower <- (1-confidence_interval)/2
  
  res <- data.frame(
    median=apply(mat,2,function(x){quantile(x,0.5)}),
    mean=apply(mat,2,function(x){mean(x)}),
    lower=apply(mat,2,function(x){quantile(x,ci_upper)}),
    upper=apply(mat,2,function(x){quantile(x,ci_lower)})
  )
}

vol <- extract_param_stan(mod,"vol") %>% 
  add_column(date=df$date,.before=1)

vol %>% 
  ggplot(aes(x=date))+theme_light()+
  geom_ribbon(aes(ymin=lower,ymax=upper),fill="lightsteelblue1",alpha=0.5)+
  geom_line(aes(y=lower),color="lightsteelblue1",alpha=0.5)+
  geom_line(aes(y=upper),color="lightsteelblue1",alpha=0.5)+
  geom_line(aes(y=median),color="firebrick")+
  scale_x_date(breaks=scales::date_breaks("1 year"),date_labels="%y")+
  labs(x="date",y="volatility")

params <- summary(mod)$summary
params <- params %>% 
  as.data.frame() %>% 
  rownames_to_column(var="param")
write.csv(params,"params.csv",row.names=F)
write.csv(vol,"vol.csv",row.names=F)


# プロット --------------------------------------------------------------------
res <- left_join(vol,df,by="date")

plot_vol <- res %>% 
  ggplot(aes(x=date))+theme_light()+
  geom_ribbon(aes(ymin=lower,ymax=upper),fill="lightsteelblue1",alpha=0.5)+
  geom_line(aes(y=lower),color="lightsteelblue1",alpha=0.5)+
  geom_line(aes(y=upper),color="lightsteelblue1",alpha=0.5)+
  geom_line(aes(y=median),color="firebrick")+
  scale_x_date(breaks=scales::date_breaks("1 year"),date_labels="%y")+
  scale_y_continuous(breaks=0:10)+
  labs(x="date",y="volatility")

plot_close <- res %>% 
  select(date,close) %>% 
  ggplot(aes(x=date,y=close))+theme_light()+geom_line()+
  scale_x_date(breaks=scales::date_breaks("1 year"),date_labels="%y")+
  labs(x="date",y="close")

patchwork::wrap_plots(plot_vol,plot_close,ncol=1)
