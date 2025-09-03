rm(list=ls())
library(tidyverse)
library(ggpubr)
library(signal)
library(here)
library(fs)
root <- path('analysis/scripts/sandbox/figures_general/cwreg_overview/')
setwd(path(here(), root))

# Import / format
dirty <- read.csv('data/dirty_timeseries.csv')
dirty$dtype <- 'dirty'
clean <- read.csv('data/clean_timeseries.csv')
clean$dtype <- 'clean'
d <- rbind(dirty, clean)

d <- d %>% 
    group_by(dtype) %>% 
    mutate(sample = 1:n()) %>% 
    ungroup() %>% 
    gather(channel, voltage, Oz:POz) %>% 
    select(dtype, sample, channel, voltage) 

qu <- read.csv('data/eeg_quality_formatted.csv')


# --- TIMESERIES PLOT --- #

# Find strong alpha
fs <- 250
# x <- d[d$dtype=='clean' & d$channel == 'Oz',]$voltage
# wt <- analyze.wavelet(
#     my.data = data.frame(signal=x),
#     my.series = 'signal',
#     loess.span = 0,
#     dt = 1/fs,
#     dj = 1/20,
#     lowerPeriod = 1/12,
#     upperPeriod = 1/8,
#     make.pval = FALSE
# )
# 
# mat <- wt$Power
# freqs <- 1 / wt$Period
# alpha <- rowMeans(t(mat))
# plot(alpha)
# 
# # Rough position of alpha spike
# idx <- 110000:115000
# alpha <- alpha[idx]

start <- 110000
twin <- start:(start + (10 * 250))

axis_text <- 10

p1 <- d %>% 
    mutate(dtype_label = ifelse(dtype=='dirty', 'Before CW regression', 
                                'After CW regression'),
           time = (sample / fs) - (start / fs)) %>% 
    mutate(dtype_label = factor(dtype_label, levels = c('Before CW regression',
                                                        'After CW regression'))) %>% 
    dplyr::filter(sample %in% twin,
                  channel == 'O2') %>% 
    ggplot(aes(x = time, y = voltage)) + 
    geom_line() +
    facet_wrap(~dtype_label, nrow = 2) + 
    labs(
        x = 'Time (s)',
        y = 'EEG potential'
    ) +
    scale_x_continuous(breaks = 0:10) + 
    theme_bw() + 
    theme(axis.ticks = element_blank(),
          panel.grid = element_blank(),
          strip.background = element_rect(fill = NA),
          axis.text.y = element_blank(),
          text = element_text(size = 16),
          axis.text = element_text(size = axis_text))
    


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
    


g <- ggarrange(p1, p2, nrow = 2, labels = c('A.', 'B.'))

ggsave(filename = 'figure1.png', plot = g, height = 6, width = 9, units = 'in', dpi = 300)


























