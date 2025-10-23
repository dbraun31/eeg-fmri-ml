# Import and drop NaNs from global
fpath <- 'DNa_ts_MSHBM_sub-001_bld002_highpass.txt'
global <- sapply(readLines(fpath), as.numeric)
global <- global[!is.nan(global)]

# Import and drop joint NaNs from rest and TP
# TP
fpath <- 'TP_DNa_ts_MSHBM_sub-001_bld002_highpass.txt'
tp <- sapply(readLines(fpath), as.numeric)

# rest  
fpath <- 'Rest_DNa_ts_MSHBM_sub-001_bld002_highpass.txt'
rest <- sapply(readLines(fpath), as.numeric)

condition <- data.frame(tp = tp, rest = rest)    

condition <- condition[!(is.nan(condition$tp) & is.nan(condition$rest)), ]

print(paste0('Rows in global data: ', length(global)))
print(paste0('Rows in condition data: ', nrow(condition)))