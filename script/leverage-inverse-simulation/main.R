library(tidyverse)
library(furrr)


JQuantsR::authorize()
topix <- JQuantsR::get_topix()
df <- topix |> 
  mutate(ret=(Close / lag(Close)) - 1) |> 
  slice(-1)
df |> head()

#' @param returns a vector of numeric: 対前日リターンのベクトル
#' @param day integer: 保有期間
#' @param multiple numeric: 比較するETFの倍率
#' @param n_replication integer: 試行回数
#' @return a vector of numeric
calc_performance <- function(returns, day, multiple, n_replication) {
  mat <- purrr::map(1:n_replication, \(x) sample(df$ret, size=day, replace=TRUE))
  
  multiple_basic <- ifelse(multiple>0, 1, -1)
  pf_basic <- mat |> 
    map_dbl(\(x) {
      cumprods <- 1*cumprod(1+x*(multiple_basic))
      cumprods[length(cumprods)]
    })
  
  pf <- mat |> 
    map_dbl(\(x) {
      cumprods <- 1*cumprod(1+x*multiple)
      cumprods[length(cumprods)]
    })
  return(pf - pf_basic)
}

n_replication <- 1000000
params <- expand.grid(days=c(5, 10, 20, 60, 120, 240), multiple=c(2, 3, -1, -2, -3)) |> 
  purrr::array_branch(margin=2)

plan(multisession)
res <- future_map2_dfr(params$days, params$multiple, function(d, m) {
  tibble(
    day=d,
    multiple=m,
    diff=calc_performance(df$ret, d, m, n_replication)
  )
}, .progress=TRUE)
plan(sequential)
# res <- map2_dfr(params$days, params$multiple, function(d, m) {
#   tibble(
#     day=d,
#     multiple=m,
#     diff=calc_performance(df$ret, d, m, n_replication)
#   )
# })

res |> 
  mutate(diff=diff*100) |> 
  group_by(multiple, day) |> 
  summarize(
    median=quantile(diff, 0.5),
    mean=mean(diff),
    q05=quantile(diff, 0.05),
    q95=quantile(diff, 0.95),
  ) |> 
  select(multiple, day, median) |> 
  pivot_wider(names_from=day, values_from=median, names_glue="{.name} days")
