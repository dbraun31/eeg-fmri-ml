rm(list=ls())
library(tidyverse)
library(ggpubr)
library(WaveletComp)
library(zoo)
library(signal)
library(here)
library(fs)
root <- path('analysis/scripts/sandbox/figures_general/cwreg_overview/')
setwd(path(here(), root))

# Import / format
pre_cw <- read.csv('data/pre_cw_timeseries.csv')
pre_cw$dtype <- 'pre_cw'
post_cw <- read.csv('data/post_cw_timeseries.csv')
post_cw$dtype <- 'post_cw'
clean <- read.csv('data/clean_timeseries.csv')
clean$dtype <- 'clean'
d <- rbind(pre_cw, post_cw, clean)

d <- d %>% 
    group_by(dtype) %>% 
    mutate(sample = 1:n()) %>% 
    ungroup() %>% 
    gather(channel, voltage, Oz:POz) %>% 
    select(dtype, sample, channel, voltage) 

qu <- read.csv('data/eeg_quality_formatted.csv')


window_size <- 10
fs <- 250
axis_text <- 10

# --- GET CHANNEL WITH HIGHEST MEAN ALPHA --- #

# Visualize channels
n_chans <- length(unique(d$channel))
d %>% 
    dplyr::filter(dtype == 'clean', sample %in% 10000:(10000 + (250))) %>% 
    ggplot(aes(x = sample, y = voltage)) + 
    geom_line() + 
    facet_wrap(~channel, nrow = n_chans)

best_channel <- d %>% 
    dplyr::filter(dtype == 'clean') %>% 
    group_by(channel) %>% 
    group_modify(~ {
        wt <- analyze.wavelet(
            my.data = data.frame(signal = .x$voltage),
            my.series = 'signal',
            loess.span = 0,
            dt = 1/250,
            dj = 1/20,
            lowerPeriod = 1/12,
            upperPeriod = 1/8,
            make.pval = FALSE
        )
        
        freqs <- 1 / wt$Period
        alpha_idx <- which(freqs >= 8 & freqs <= 12)
        mean_alpha <- mean(wt$Power[alpha_idx,])
        data.frame(mean_alpha = mean_alpha)
    }) %>% 
    ungroup() %>% 
    print() %>% 
    dplyr::filter(mean_alpha == max(mean_alpha)) %>% 
    pull(channel)

# --- FIND PEAK 10 S ALPHA IN BEST CHANNEL --- #

x <- d[d$dtype=='clean' & d$channel==best_channel,]
wt <- analyze.wavelet(
    my.data = x,
    my.series = 'voltage',
    loess.span = 0,
    dt = 1/250,
    dj = 1/20,
    lowerPeriod = 1/12,
    upperPeriod = 1/8,
    make.pval = FALSE
)

alpha <- colMeans(wt$Power)
pow <- data.frame(sample = 1:length(alpha), alpha = alpha)

ma <- rollmedian(alpha, window_size * fs, fill = NA)
ma <- data.frame(sample = 1:(length(ma)), alpha = ma)
best_sample <- ma %>% 
    dplyr::filter(sample >= 10000 & sample <= 150000) %>% 
    dplyr::filter(alpha == max(alpha)) %>% 
    pull(sample)
    
best_sample <- median(best_sample)


start <- best_sample - (window_size/2 * fs)
t_win <- start:(start + window_size*fs)

# --- COMPUTE TIME FREQUENCY --- #

# Find strong alpha
x <- d[d$dtype=='post_cw' & d$channel == best_channel,]$voltage
post_cw_waves <- analyze.wavelet(
    my.data = data.frame(signal=x),
    my.series = 'signal',
    loess.span = 0,
    dt = 1/fs,
    dj = 1/20,
    lowerPeriod = 1/40,
    upperPeriod = 1/1,
    make.pval = FALSE
)
post_cw_mat <- post_cw_waves$Power

x <- d[d$dtype=='pre_cw' & d$channel == best_channel,]$voltage
pre_cw_waves <- analyze.wavelet(
    my.data = data.frame(signal=x),
    my.series = 'signal',
    loess.span = 0,
    dt = 1/fs,
    dj = 1/20,
    lowerPeriod = 1/40,
    upperPeriod = 1/1,
    make.pval = FALSE
)
pre_cw_mat <- pre_cw_waves$Power

x <- d[d$dtype=='clean' & d$channel == best_channel,]$voltage
clean_waves <- analyze.wavelet(
    my.data = data.frame(signal=x),
    my.series = 'signal',
    loess.span = 0,
    dt = 1/fs,
    dj = 1/20,
    lowerPeriod = 1/40,
    upperPeriod = 1/1,
    make.pval = FALSE
)
clean_mat <- clean_waves$Power

# format
post_cw_df <- data.frame(t(post_cw_mat))
colnames(post_cw_df) <- as.character(post_cw_waves$Period)
post_cw_df$dtype <- 'post_cw'
pre_cw_df <- data.frame(t(pre_cw_mat))
colnames(pre_cw_df) <- as.character(pre_cw_waves$Period)
pre_cw_df$dtype <- 'pre_cw'
clean_df <- data.frame(t(clean_mat))
colnames(clean_df) <- as.character(clean_waves$Period)
clean_df$dtype <- 'clean'

waves <- rbind(post_cw_df, pre_cw_df, clean_df)
waves <- waves %>% 
    group_by(dtype) %>% 
    mutate(sample = 1:n()) %>% 
    ungroup() %>% 
    gather(period, power, -sample, -dtype) %>% 
    mutate(period = as.numeric(period)) %>% 
    mutate(frequency = 1 / period) %>% 
    select(-period)


# --- FIGURE: INTERLEAVED TIME DOMAIN - TIME-FREQ DOMAIN ---  #

p1 <- d %>% 
    dplyr::filter(dtype == 'pre_cw', channel == 'O2', sample %in% t_win) %>% 
    mutate(time = sample / fs) %>% 
    mutate(time = time - min(time)) %>% 
    ggplot(aes(x = time, y = voltage)) + 
    geom_line() + 
    labs(
        x = 'Time (s)',
        y = 'EEG potential (V)',
        title = 'Before CW regression'
    ) + 
    ylim(-8e-05, 7e-5) + 
    theme_bw() + 
    theme(axis.ticks = element_blank(),
          panel.grid = element_blank(),
          text = element_text(size = 16),
          axis.text = element_text(size = axis_text))
        
max_p <- ceiling(max(waves[waves$sample %in% t_win,]$power))
p2 <- waves %>% 
    dplyr::filter(dtype=='pre_cw', sample %in% t_win) %>% 
    mutate(time = sample / fs) %>% 
    mutate(time = time - min(time)) %>% 
    ggplot(aes(x = time, y = frequency, fill = power)) + 
    geom_raster(interpolate=TRUE) + 
    scale_y_log10() + 
    scale_fill_viridis_c(option = 'plasma', limits = c(0, max_p), breaks = seq(0, max_p, length.out=2)) + 
    labs(
        x = 'Time (s)',
        y = 'Frequency (Hz)',
        fill = 'Power'
    ) + 
    theme_bw() + 
    theme(axis.ticks = element_blank(),
          panel.grid = element_blank(),
          text = element_text(size = 16),
          axis.text = element_text(size = axis_text),
          legend.position = 'none')

p3 <- d %>% 
    dplyr::filter(dtype == 'post_cw', channel == 'O2', sample %in% t_win) %>% 
    mutate(time = sample / fs) %>% 
    mutate(time = time - min(time)) %>% 
    ggplot(aes(x = time, y = voltage)) + 
    geom_line() + 
    labs(
        x = 'Time (s)',
        y = 'EEG potential (V)',
        title = 'After CW regression'
    ) + 
    ylim(-8e-05, 7e-5) + 
    theme_bw() + 
    theme(axis.ticks = element_blank(),
          panel.grid = element_blank(),
          text = element_text(size = 16),
          axis.text = element_text(size = axis_text))

p4 <- waves %>% 
    dplyr::filter(dtype=='post_cw', sample %in% t_win) %>% 
    mutate(time = sample / fs) %>% 
    mutate(time = time - min(time)) %>% 
    ggplot(aes(x = time, y = frequency, fill = power)) + 
    geom_raster(interpolate=TRUE) + 
    scale_y_log10() + 
    scale_fill_viridis_c(option = 'plasma', limits = c(0, max_p), breaks = seq(0, max_p, length.out=2)) + 
    labs(
        x = 'Time (s)',
        y = 'Frequency (Hz)',
        fill = 'Power'
    ) + 
    theme_bw() + 
    theme(axis.ticks = element_blank(),
          panel.grid = element_blank(),
          text = element_text(size = 16),
          axis.text = element_text(size = axis_text),
          legend.position = 'none')

p5 <- d %>% 
    dplyr::filter(dtype == 'clean', channel == 'O2', sample %in% t_win) %>% 
    mutate(time = sample / fs) %>% 
    mutate(time = time - min(time)) %>% 
    ggplot(aes(x = time, y = voltage)) + 
    geom_line() + 
    labs(
        x = 'Time (s)',
        y = 'EEG potential (V)',
        title = 'After full preprocessing'
    ) + 
    ylim(-8e-05, 7e-5) + 
    theme_bw() + 
    theme(axis.ticks = element_blank(),
          panel.grid = element_blank(),
          text = element_text(size = 16),
          axis.text = element_text(size = axis_text))

p6 <- waves %>% 
    dplyr::filter(dtype=='clean', sample %in% t_win) %>% 
    mutate(time = sample / fs) %>% 
    mutate(time = time - min(time)) %>% 
    ggplot(aes(x = time, y = frequency, fill = power)) + 
    geom_raster(interpolate=TRUE) + 
    scale_y_log10() + 
    scale_fill_viridis_c(option = 'plasma', limits = c(0, max_p), breaks = seq(0, max_p, length.out=2)) + 
    labs(
        x = 'Time (s)',
        y = 'Frequency (Hz)',
        fill = 'Power'
    ) + 
    theme_bw() + 
    theme(axis.ticks = element_blank(),
          panel.grid = element_blank(),
          text = element_text(size = 16),
          axis.text = element_text(size = axis_text),
          legend.position = 'none')

g <- ggarrange(ggarrange(p1, p2, nrow = 2), ggarrange(p3, p4, nrow = 2), 
               ggarrange(p5, p6, nrow = 2), nrow = 3, labels = c('A.', 'B.', 'C.'))

ggsave(filename='r01_figure.png', plot = g, height = 14, width = 10, units = 'in', dpi = 300)








# temp - zoom in

seq_log <- exp(seq(log(1), log(40), length.out = 10))

waves %>% 
    dplyr::filter(dtype=='clean', sample %in% t_win) %>% 
    mutate(time = sample / fs) %>% 
    mutate(time = time - min(time)) %>% 
    dplyr::filter(time <= 5) %>% 
    ggplot(aes(x = time, y = frequency, fill = power)) + 
    geom_raster(interpolate=TRUE) + 
    scale_y_log10(breaks = seq_log) + 
    scale_fill_viridis_c(option = 'plasma', limits = c(0, 250), breaks = seq(0, 250, length.out=2)) + 
    labs(
        x = 'Time (s)',
        y = 'Frequency (Hz)',
        fill = 'Power'
    ) + 
    theme_bw() + 
    theme(axis.ticks = element_blank(),
          panel.grid = element_blank(),
          text = element_text(size = 16),
          axis.text = element_text(size = axis_text),
          legend.position = 'none')


ggsave('zoom.png', height = 1080, width = 1920, units = 'px', dpi = 200)













# --- HISTOGRAMS --- #

p2 <- qu %>% 
    gather(metric, value, good_chans:brain_ica) %>% 
    group_by(subject, session, metric) %>% 
    summarize(value = mean(value)) %>% 
    mutate(metric = recode(metric, `brain_ica` = 'ICA components\nlabeled brain',
                           `good_chans` = 'Good channels',
                           `good_data` = 'Good data')) %>% 
    ggplot(aes(x = value)) + 
    geom_histogram(fill = '#0077BB', color = 'white', bins = 20) + 
    labs(
        x = 'Proportion',
        y = 'Frequency'
    ) + 
    facet_wrap(~metric, nrow = 1, scales = 'free_y')  +
    xlim(0, 1) + 
    theme_bw() + 
    theme(axis.ticks = element_blank(),
          panel.grid = element_blank(),
          strip.background = element_rect(fill = NA),
          text = element_text(size = 16),
          axis.text = element_text(size = axis_text))
    