library(arrow)
library(tidyverse)
library(data.table)
library(here)
library(fs)
setwd(path(here(), 'analysis'))
root <- path('scripts/sandbox/correlations')

d <- data.table(read_feather('data/merged_data.feather'))

esub <- d[d$subject == 'sub-001' & !is.na(d$dmn),]

# Getting NAs here and idk why
cors <- sapply(voltage_cols, FUN = function(x) cor(esub$dmn, esub[,x]))

voltage_cols <- colnames(d)[6:length(colnames(d))]

# Big data table energy
result <- d[!is.na(dmn),
  .(cors = list(sapply(.SD, function(col) cor(dmn, col, use = 'pairwise.complete.obs')))),
  by = .(subject, session, run),
  .SDcols = voltage_cols
  ][
      ,
      .(mean_cors = list(Reduce(`+`, cors) / length(cors))),
      by = subject
  ][
      ,
      .(feature = voltage_cols, mean_cors = unlist(mean_cors)),
      by = subject
  ]


stat_dat <- result %>% 
    group_by(subject) %>% 
    summarize(ci_h = quantile(mean_cors, probs = .975),
           ci_l = quantile(mean_cors, probs = .025)) 

result %>% 
    ggplot(aes(x = mean_cors)) + 
    geom_vline(xintercept = 0, linetype = 'dashed') + 
    geom_vline(data = stat_dat, aes(xintercept = ci_h), linetype = 'dashed', color = 'orange') + 
    geom_vline(data = stat_dat, aes(xintercept = ci_l), linetype = 'dashed', color = 'orange') + 
    geom_histogram(color = 'black', fill = 'steelblue', alpha = .6) + 
    facet_wrap(~subject) + 
    xlim(-.5, .5) + 
    labs(
        x = 'Correlations between features and DMN',
        y = 'Frequency'
    ) + 
    theme_bw() + 
    theme(panel.grid = element_blank(),
          axis.ticks = element_blank(),
          strip.background = element_rect(fill = NA),
          text = element_text(size = 16),
          axis.text.x = element_text(size = 12)
          )


# Screens
ggsave(path(root, 'figures/correlation_histograms.png'), width = 1920, height = 1080, units = 'px', dpi = 120)
