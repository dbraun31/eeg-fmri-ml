rm(list=ls())
library(tidyverse)
library(data.table)
library(paletteer)
library(future)
library(future.apply)
library(glue)
library(psych)
library(here)
library(fs)
library(arrow)


get_correlations <- function(script_root, runs=1:4) {
    # Whether to compute correlations across different sets of runs
    # Saves out to cache
    # (very RAM intensive)
    source(path(here(), 'analysis/scripts/modules/preprocessing/04a-compute_correlations_functions.r'))
    
    d <- read_feather(path(data_root, '../correlation_data/merged_data.feather'))
    
    min_trs <- d %>% 
        select(subject:tr) %>% 
        group_by(subject, session, run) %>% 
        summarize(tr = max(tr)) %>% 
        group_by(session, run) %>% 
        summarize(tr_minimum = min(tr)) 
    
    # Filter down to minimum TRs per run
    d <- d %>% 
        inner_join(min_trs) %>% 
        filter(tr <= tr_minimum) %>% 
        select(-tr_minimum)
    
    cache_dir <- path(script_root, 'cache')
    if (!dir.exists(cache_dir)) {
        dir_create(cache_dir)
    }
    
        preprocess_root <- path(here(), 'analysis/scripts/modules/preprocessing')
        ch_names <- suppressWarnings(readLines(path(data_root, '../correlation_data/ch_names.txt')))
        
        lapply(runs, FUN = function(run_int) {
            print(glue('Calculating subset {run_int} of {max(runs)}'))
            print('')
            run_list <- paste0('run-00', 1:run_int)
            d_sub <- d[d$run %in% run_list,]
            cors <- correlation_pipeline(d_sub, preprocess_root, ch_names)
            cors$run_set <- run_int
            write_feather(cors, path(cache_dir, glue('cache_run-00{run_int}.feather')))
            rm(cors)
            gc()
            NULL
        })
}

get_iccs <- function(script_root) {
    
    caches <- list.files(path(script_root, 'cache'), full.names=TRUE)
    print('Loading in data...')
    d <- do.call(rbind, lapply(caches, read_feather))
    d <- data.table(d)
    # Drop subject 3 (no session 1 data)
    d <- d[subject != 'sub-003']
    
    ex <- dcast(d[channel == 'Fp1' & frequency == 1 & lag == 0 & network == 'dATNa' & run_set==4, 
              .(cors = mean(cors)), by = .(subject, session)],
          subject ~ session,
          value.var = 'cors')[, subject := NULL]
    
    
    d[, feature_set := paste(channel, frequency, lag, sep = '_')]
    
    feature_sets <- unique(d$feature_set)
    
    d_run_set <- d[run_set == 4]
    
    apply_icc <- function(d_run_set) {
        
        # Set up parallel
        max_workers <- 1000
        workers = max(1, min(parallel::detectCores() - 1, max_workers))
        if (interactive()) {
            plan('multisession', workers = 1)
        } else {
            plan('multicore', workers=workers)
        }
        options(future.globals.maxSize = 16 * 1024^3)
        
        out <- future_sapply(feature_sets, FUN = function(x, d_run_set) {
            # Apply ICC to each feature set
            d_sub <- d_run_set[feature_set == x]
            ses_long <- d_sub[, .(cors = mean(cors, na.rm=TRUE)), by = .(subject, session)]
            ses_wide <- dcast(ses_long, subject ~ session, value.var='cors')[, subject := NULL]
            icc <- suppressWarnings(suppressMessages(ICC(ses_wide)$result))
            icc <- icc[icc$type == 'ICC3',]$ICC
            icc
        }, d_run_set = d_run_set)
        
        return(out)
    }
    
    # Apply ICC to each run set
    run_sets <- unique(d$run_set)
    print('Computing ICCs...')
    result <- lapply(run_sets, FUN = function(rs) apply_icc(d[run_set==rs]))
    names(result) <- paste0('run_set_', run_sets)
    
    return(result)
    
}


format_iccs <- function(iccs) {
    format_run_set <- function(run_set, label) {
        ds <- data.frame(feature = names(run_set), icc = unname(run_set))
        ds %>% 
            separate(feature, into = c('channel', 'frequency', 'lag'), sep = '_') %>% 
            mutate(run_set = label)
    }
    
    out <- do.call(rbind, lapply(names(iccs), FUN = function(x) format_run_set(iccs[[x]], label=x)))
    rownames(out) <- NULL
    return(out)
}






# -- SCRIPT STARTS EXECUTING HERE -- #

args <- commandArgs(trailingOnly=TRUE)

if (length(args) == 0) {
    setwd(here())
    data_root <- path('analysis/data/original')
} else {
    data_root <- NULL
    for (arg in args) {
        if (grepl('/', arg)) {
            data_root <- path(arg)
        } else {
            runs <- as.integer(arg)
        }
    }
    if (is.null(data_root)) {
        data_root <- path('analysis/data/original')
        setwd(here())
    }
}

script_root <- path(here(), 'analysis/scripts/sandbox/session_icc_steps')


compute_correlations <- FALSE
if (compute_correlations) {
    get_correlations(script_root, runs=runs)
}

compute_iccs <- TRUE
if (compute_iccs) {
    iccs <- get_iccs(script_root)
    dpath <- path(data_root, '../correlation_data/iccs.csv')
    d <- format_iccs(iccs)
    write.csv(d, dpath, row.names=FALSE)
    cache_dir <- path(script_root, 'cache')
    if (dir_exists(cache_dir)) {
        dir_delete(cache_dir)
    }
}









