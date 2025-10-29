rm(list=ls())
library(reticulate)
library(tidyverse)
library(here)
library(glue)
library(ggpubr)
library(fs)
fig_save_root <- path(here(), 'analysis/scripts/sandbox/figures/figures_scratch')
overall_text <- 18
script_root <- as.character(path(here(), 'analysis/scripts/sandbox/figures/eeg_timeseries'))

use_condaenv('eeg-fmri')
py_run_string('
import numpy as np
raw = np.load(f"{r.script_root}/timeseries.npy")
power = np.load(f"{r.script_root}/power.npy")
')

ch_names <- suppressWarnings(readLines(path(script_root, 'ch_names.txt')))

raw <- py$raw
power <- py$power

# Convert to long
dimnames(power) <- list(
    sample = 1:dim(power)[1],
    channel = ch_names,
    freq = 1:40
)

power <- as.data.frame.table(power, responseName = 'power')


set.seed(42)
random_channel <- sample(ch_names, size = 1)

n_chans <- length(ch_names) %/% 3
random_channels <- sample(ch_names, size = n_chans)
if (!random_channel %in% random_channels) {
    random_channels <- c(random_channels[1:(length(random_channels)-1)], random_channel)
}


# Timeseries plot
dimnames(raw) <- list(
    sample = 1:dim(raw)[1],
    channel = ch_names
)

raw <- as.data.frame.table(raw, responseName = 'voltage')
raw <- raw %>% 
    mutate(sample = as.numeric(sample)) %>% 
    mutate(time = (sample - min(sample)) / 250)
 
time_min <- 0
time_max <- ceiling(max(raw$time))
time_mid <- as.integer((time_min + time_max) / 2)

# Normalize
raw <- raw %>% 
    group_by(channel) %>% 
    mutate(voltage_n = scale(voltage)[,1])

p1 <- raw %>% 
    filter(channel %in% random_channels) %>% 
    mutate(channel = factor(channel, levels = ch_names[order(ch_names)])) %>% 
    ggplot(aes(x = time, y = voltage_n, group = 1)) + 
    geom_line() + 
    facet_grid(channel~.) + 
    labs(
        x = 'Time (s)',
        y = 'EEG Voltage'
    ) + 
    scale_x_continuous(
        breaks = c(time_min, time_mid, time_max),
        labels = function(x) as.integer(x)
    ) + 
    theme_bw() + 
    theme(axis.text.y = element_blank(),
          panel.grid = element_blank(),
          axis.ticks = element_blank(),
          strip.background = element_rect(fill = NA, color = 'black'),
          text = element_text(size = overall_text))
    

# TF plot
power <- power %>% 
    mutate(sample = as.numeric(sample),
           freq = as.numeric(as.character(freq))) %>% 
    mutate(time = (sample - min(sample)) / 250)

green <- paletteer::paletteer_c('ggthemes::Green', n=100)

clip_quantile <- .95
p2 <- power %>% 
    filter(channel == random_channel) %>% 
    mutate(power = ifelse(power > quantile(power, clip_quantile), quantile(power, clip_quantile), power)) %>% 
    ggplot(aes(x = time, y = freq)) + 
    geom_raster(aes(fill = power), interpolate = TRUE) +
    labs(
        x = 'Time (s)',
        y = 'Frequency (Hz)',
        fill = 'Power',
        caption = glue('Channel: {random_channel}')
    ) + 
    scale_fill_gradientn(colors = green) + #, limits = c(quantile(power$power, 0.05), quantile(power$power, 0.95))) + 
    scale_x_continuous(breaks = c(time_min, time_mid, time_max),
                       labels = function(x) as.integer(x)) +
    theme_bw() + 
    theme(panel.grid = element_blank(),
          axis.ticks = element_blank(),
          legend.position = 'bottom',
          legend.text = element_text(angle = 45, hjust = 1, size = 10),
          legend.title = element_text(size = 9),
          legend.key.size = unit(.5, 'cm'),
          text = element_text(size = overall_text))
    
fill <- ggplot() + 
    theme_void() + 
    theme(panel.background = element_rect(fill = 'white'))
    
g <- ggarrange(p1, fill, p2, nrow = 1, widths = c(1, .1, 1))    
    
ggsave(plot=g, file = path(fig_save_root, 'eeg_timeseries.png'),
       width = 12, height = 6, units = 'in', dpi = 300)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
