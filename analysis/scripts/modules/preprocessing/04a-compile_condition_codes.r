
# This needs thorough commenting
# Skipping subject 23 session 2 for now
# (need to go back to the bids data and look at events.tsv)

parse_run <- function(combo, data_root) {
    subject <- combo$subject
    session <- combo$session
    network <- combo$network
    dir_path <- file.path(data_root, subject, session, 'func')
    
    # Import gradcpt
    pattern <- glue('{network}_.*{subject}_bld001.*')
    file <- list.files(dir_path, pattern = pattern, full.names = TRUE)
    fmri <- sapply(readLines(file), as.numeric)
    gradcpt <- data.frame(run = 'run-001', tr = 1:length(fmri), condition = 'gradcpt', fmri = fmri)
    
    task_dfs <- list()
    # Import experience sampling / rest
    for (task in c('TP', 'Rest')) {
    
        pattern <- glue('{task}_{network}.*{subject}.*')
        files <- list.files(dir_path, pattern=pattern, full.names = TRUE)
        
        task_dfs[[task]] <- do.call(rbind, lapply(files, FUN = function(x) {
            # Make run string with three zero pads
            run_int <- as.integer(str_extract(x, 'bld(\\d+)', group = 1))
            run <- sapply(run_int, FUN = function(x) glue('run-00{x}'))
            fmri = sapply(readLines(x), as.numeric)
            d <- data.frame(run = run, tr = 1:length(fmri)) 
            d[[glue('{task}_fmri')]] <- fmri
            d
        }))
    }
    
    
    # -- Format task dfs -- 
    
    # Gradcpt
    gradcpt <- gradcpt %>% 
        mutate(condition =  ifelse(is.nan(fmri), 'head_motion', 'gradcpt')) %>% 
        select(-fmri) 
    
    # Experience sampling
    es <- cbind(task_dfs[[1]], rest_fmri = task_dfs[[2]][,'Rest_fmri'])
    
    es <- es %>% 
        mutate(condition = case_when(
            is.nan(TP_fmri) & is.nan(rest_fmri) ~ 'head_motion',
            is.nan(TP_fmri) & !is.nan(rest_fmri) ~ 'rest',
            !is.nan(TP_fmri) & is.nan(rest_fmri) ~ 'thought_probe',
            .default = '-99'
        )) %>% 
        select(-TP_fmri, -rest_fmri) 
    
    out <- rbind(gradcpt, es)
    
    # Drop first lags
    num_lags <- suppressWarnings(as.integer(readLines(path(data_root, '../formatted/num_lags.txt'))))
    out <- out[num_lags:nrow(out),]
    
    # Drop head motion
    out <- out[out$condition != 'head_motion',]
    
    stem <- data.frame(subject=subject, session=session)    
    return(cbind(stem, out))
    
}

get_condition_codes <- function(data_root, network) {
    
    if (data_root == path('analysis/data/original')) setwd(here())
    subjects <- list.dirs(path(data_root), recursive=FALSE, full.names = FALSE)
    sessions <- c('ses-001', 'ses-002')
    
    combos <- expand.grid(subject=subjects, session=sessions,  network=network)
    # Yank sub 23 ses 002 for now
    combos <- combos[!(combos$subject == 'sub-023' & combos$session == 'ses-002'),]
    
    d <- do.call(rbind, lapply(1:nrow(combos), FUN=function(x) parse_run(combos[x, ], data_root)))
    
    return(d)
}


if (FALSE) {

    # sanity checks
    
    cc <- condition_codes 
    cc_s <- cc %>% 
        filter(subject == 'sub-001') %>% 
        group_by(session, run) %>% 
        summarize(count = n())
    
    
    d_s <- d %>% 
        filter(subject == 'sub-001') %>% 
        group_by(session, run) %>% 
        summarize(count = n())
    
    fpath <- 'analysis/data/original/sub-001/ses-001/func/DNa_ts_MSHBM_sub-001_bld002_highpass.txt'
    global <- sapply(readLines(fpath), as.numeric)
    
    # tp
    fpath <- 'analysis/data/original/sub-001/ses-001/func/TP_DNa_ts_MSHBM_sub-001_bld002_highpass.txt'
    tp <- sapply(readLines(fpath), as.numeric)
    
    # rest  
    fpath <- 'analysis/data/original/sub-001/ses-001/func/Rest_DNa_ts_MSHBM_sub-001_bld002_highpass.txt'
    rest <- sapply(readLines(fpath), as.numeric)
    
    condition <- data.frame(tp = tp, rest = rest)    
    
    condition <- condition[!(is.nan(condition$tp) & is.nan(condition$rest)), ]
    
    
    


}