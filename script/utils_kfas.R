grid_search_KFAS <- function(model,range_grids,num,parallel=TRUE) {
  grids <- expand.grid(replicate(num,range_grids,simplify=FALSE)) %>% 
    purrr::array_tree(margin=1)
  if (parallel) {
    future::plan(future::multisession)
    optims <- grids %>% 
      furrr::future_map_dfr(~{
        fit <- fitSSM(model,inits=.x,method="BFGS")
        tibble(
          params=list(.x),
          par=list(fit$optim.out$par),
          value=fit$optim.out$value,
          convergence=fit$optim.out$convergence
        )
      },.progress=TRUE)
    future::plan(future::sequential)
  } else {
    optims <- grids %>% 
      purrr::map2_dfr(1:length(.),~{
        cat(.y,"\n")
        fit <- fitSSM(model,inits=.x,method="BFGS")
        tibble(
          params=list(.x),
          par=list(fit$optim.out$par),
          value=fit$optim.out$value,
          convergence=fit$optim.out$convergence
        )
      },.progress=TRUE)
  }
  return(optims)
}


# ベータの値を取り出す
# フィルタ化推定量のbeta_filteredの推定値と信頼区間は以下と一致する（野村2章）
# afilt <- kfs$a[-1]; Pfilt <- kfs$P[,,-1] - fit$model$Q
# afiltconf <- cbind(afilt+sqrt(Pfilt)*qnorm(0.025),afilt+sqrt(Pfilt)*qnorm(0.975))
# kfs$aはnumeric型だがkfs$attやkfs$alphahatはts型なので、後者はas.numericする必要がある（しないとpivot_longerでエラーが出る）
# predict(fit$model,interval="confidence",filtered=T)はベータの推定値ではない（smoothedの場合、filteredならfiltered=Tにする）
extract_param <- function(kfs,param_name,date,confidence_interval=0.95) {
  idx_param_of_std_error <- which(colnames(kfs$att)==param_name)
  upper <- confidence_interval+(1-confidence_interval)/2
  lower <- (1-confidence_interval)/2
  df <- data.frame(
    filtered=as.numeric(kfs$att[,param_name]),
    std_error_filtered=sqrt(kfs$Ptt[idx_param_of_std_error,idx_param_of_std_error,]),
    smoothed=as.numeric(kfs$alphahat[,param_name]),
    std_error_smoothed=sqrt(kfs$V[idx_param_of_std_error,idx_param_of_std_error,])
  ) %>% 
    add_column(date=date,.before=1) %>% 
    mutate(
      filtered_upper=filtered+qnorm(upper)*std_error_filtered,
      filtered_lower=filtered+qnorm(lower)*std_error_filtered,
      smoothed_upper=smoothed+qnorm(upper)*std_error_smoothed,
      smoothed_lower=smoothed+qnorm(lower)*std_error_smoothed
    )
}
