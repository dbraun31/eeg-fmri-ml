rm(list=ls())
library(tidyverse)
library(paletteer)
library(arrow)
library(fs)
library(here)


# --- PLOT CONFIG VARS --- #
# (you can change any of these before outputting a specific plot and it'll update for that plot)
# (or keep them constant and they'll apply to all corresponding plots)

overall_text <- 18
axis_text <- 16
gradient_colors <- paletteer_c('ggthemes::Green', n=100)


# ----------------------- #

setwd(here())
data_root <- path('analysis/data/original')
fig_save_root <- path('analysis/scripts/sandbox/session_icc_steps')

# Add path to data and figure save root
# data_root <- path('path/to/data')
# fig_save_root <- path('path/to/figures)

# Full data for TRs in run histograms across subjects
d_full <- read_feather(path(data_root, '../correlation_data/merged_data.feather'))

dpath <- path(data_root, '../correlation_data/iccs.csv')
d <- read.csv(dpath)

# Inspect TRs by run
d_full %>% 
    select(subject:tr) %>% 
    group_by(subject, session, run) %>% 
    summarize(tr = max(tr)) %>% 
    mutate(mins = tr * 2 / 60) %>% 
    ggplot(aes(x = mins)) + 
    geom_histogram(fill = 'steelblue', color = 'black') +
    facet_grid(session~run) + 
    labs(
        x = 'Run length (min)',
        y = 'Frequency'
    ) + 
    theme_bw() + 
    theme(panel.grid = element_blank(),
          axis.ticks = element_blank(),
          strip.background = element_rect(fill = NA, color = 'black'),
          text = element_text(size = overall_text),
          axis.text = element_text(size = axis_text))

ggsave(path(fig_save_root, 'mins_per_run.png'), width = 1920, height = 1080, units = 'px', dpi = 150)

rm(d_full)
gc()

# Visualize aggregate result
d %>% 
    mutate(run = as.integer(str_extract(run_set, '_(\\d)', group = 1))) %>% 
    group_by(run) %>% 
    summarize(icc = mean(icc)) %>% 
    ggplot(aes(x = run, y = icc)) +
    geom_point() + 
    geom_line() + 
    ylim(0, .2) + 
    labs(
        x = 'Number of runs in data',
        y = 'Mean intraclass correlation coefficient'
    ) + 
    theme_bw() + 
    theme(axis.ticks = element_blank(),
          panel.grid = element_blank(),
          text = element_text(size = overall_text),
          axis.text = element_text(size = axis_text))

ggsave(path(fig_save_root, 'icc_overall.png'), width = 1920, height = 1080, units = 'px', dpi = 150)

greens <- paletteer_c('ggthemes::Green', 100)

d %>% 
    mutate(run_label = recode(run_set, `run_set_1` = 'One run',
                              `run_set_2` = 'Two runs',
                              `run_set_3` = 'Three runs',
                              `run_set_4` = 'Four runs'),
           lag = as.numeric(lag) * 2,
           frequency = as.numeric(frequency)) %>% 
    mutate(run_label = factor(run_label, levels = c('One run', 'Two runs', 
                                                    'Three runs', 'Four runs'))) %>% 
    group_by(frequency, lag, run_label) %>% 
    summarize(icc = mean(icc)) %>% 
    ggplot(aes(x = frequency, y = lag)) + 
    geom_tile(aes(fill = icc)) + 
    facet_wrap(~run_label) +
    scale_fill_gradientn(colors = gradient_colors) + 
    labs(
        x = 'Frequency (Hz)',
        y = 'Lag (s)',
        fill = 'Intraclass correlation\ncoefficient'
    ) + 
    theme_bw() + 
    theme(panel.grid = element_blank(),
          axis.ticks = element_blank(),
          strip.background = element_rect(fill = NA, color = 'black'),
          legend.position = 'bottom',
          legend.text = element_text(angle = 45, hjust = 1),
          text = element_text(size = overall_text),
          axis.text = element_text(size = axis_text))


ggsave(path(fig_save_root, 'icc_freq_lag.png'), width = 1920, height = 1080, units = 'px', dpi = 150)
