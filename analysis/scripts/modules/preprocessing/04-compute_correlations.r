rm(list=ls())
library(tidyverse)
library(data.table)
library(future.apply)
library(here)
library(fs)
library(arrow)
setwd(here())
root <- path('analysis/scripts/sandbox/correlations')

args <- commandArgs(trailingOnly=TRUE)

if (length(args) == 0) {
    data_root <- path('analysis/data')
} else{
    data_root <- path(args[1])
}

# Function defs

extract_matrices <- function(run_df, eeg_cols, fmri_cols) {
    eeg_mat <- as.matrix(run_df[, eeg_cols])
    fmri_mat <- as.matrix(run_df[, fmri_cols])
    return(list(eeg=eeg_mat, fmri=fmri_mat))
}

compute_spearman_cor <- function(eeg, fmri) {
    # Apply column-wise rank
    eeg_rank <- apply(eeg, 2, rank)
    fmri_rank <- apply(fmri, 2, rank)
    
    # Normalize
    eeg_z <- scale(eeg_rank)
    fmri_z <- scale(fmri_rank)
    n_trs <- nrow(eeg)
    
    # Correlate
    cor <- t(fmri_z) %*% eeg_z / (n_trs - 1)
    
    return(cor)
    
}


# Run 

if(file.exists(path(root, 'correlation_data/cors.rds'))){
    cors <- readRDS(path(root, 'correlation_data/cors.rds'))
} else {
    
    # Import
    if (!file.exists(path(data_root, 'correlation_data/merged_data.feather'))) {
        stop('Need to run 03-make_flat_data.py before this script.')
    }
    d <- read_feather(path(data_root, 'correlation_data/merged_data.feather'))
    
    # Extract modality columns
    eeg_cols <- colnames(d)[which(colnames(d) == 'Fp1_1_0'):(ncol(d))]
    fmri_cols <- colnames(d)[(which(colnames(d) == 'tr')+1):(which(colnames(d) == 'Fp1_1_0')-1)]
    
    # Split data by subject, session, run
    d$run_id <- paste(d$subject, d$session, d$run, sep = '_')
    d_split <- split(d, d$run_id)
    
    
    # Test it out
    run_matrices <- extract_matrices(d_split[[1]], eeg_cols, fmri_cols)
    cor_mat <- compute_spearman_cor(run_matrices$eeg, run_matrices$fmri)
    plan(multisession, workers=parallel::detectCores()-5)
    
    cors <- future_lapply(d_split, function(run_df) {
        mats <- extract_matrices(run_df, eeg_cols, fmri_cols)
        compute_spearman_cor(mats$eeg, mats$fmri)
    })
    
    
    saveRDS(cors, file = path(root, 'cors.rds'))
}

format_cors <- function(data, label) {
    data <- data.frame(t(data))
    info <- str_split(label, '_')[[1]]
    subject <- info[1]
    session <- info[2]
    run <- info[3]
    
    header <- data.frame(subject=subject, session=session, run=run, eeg_feature=rownames(data))
    out <- cbind(header, data)
}


ch_names <- readRDS(path(root, 'ch_names.rds'))

# Combine
d <- do.call(rbind, lapply(seq_along(cors), function(i) format_cors(cors[[i]], names(cors[i]))))
d <- d %>% 
    separate(eeg_feature, into = c('channel', 'frequency', 'lag'), sep = '_') %>%
    mutate(frequency = as.numeric(frequency),
           lag = as.numeric(lag),
           channel = factor(channel, levels = ch_names))

fmri_cols <- colnames(d)[(which(colnames(d) == 'lag')+1):(ncol(d))]

d <- d %>% 
    gather(network, cors, all_of(fmri_cols)) %>% 
    mutate(network = recode(network, `FPCNa` = 'FPNa', `FPCNb` = 'FPNb',
                            `DAN` = 'dATN', `DANa` = 'dATN-A', `DANb` = 'dATN-B'))

# Save
write_feather(d, path(root, 'correlations_long.feather'))







