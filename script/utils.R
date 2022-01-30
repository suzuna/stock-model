read_stockcsv_daily <- function(file,ID) {
  data <- read_csv(file,col_types="ccddddddd",skip=1,locale=locale(encoding="Shift-JIS")) %>% 
    set_names(c("date","time","open","high","low","close","volume","oi"))
  data <- data %>% 
    mutate(
      ID=ID,
      date=as.Date(date,tz="Asia/Tokyo"),
      datetime=lubridate::ymd_hms(str_c(date,time,sep=" "),tz="Asia/Tokyo")
    ) %>% 
    arrange(date) %>% 
    mutate(ret=(log(close)-log(lag(close)))*100) %>% 
    select(ID,date,open,high,low,close,volume,ret)
}
