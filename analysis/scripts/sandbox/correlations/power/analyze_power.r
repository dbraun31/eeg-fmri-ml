# --- POWER ANALYSIS --- #
# Analyze power for both regular and ICC correlations
# Across DNa, DNb, DAN, and FPNa/FPNb

rm(list=ls())
library(tidyverse)
library(here)
library(pwr)
library(fs)
library(glue)
library(arrow)
setwd(here())
root <- path('analysis/scripts/sandbox/correlations/power')

breaks <- c(0, 1, 4, 8, 12, 30, 40)
labels <- c('init', 'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma')
bins <- unique(cut(d$frequency, breaks=breaks))
labels <- paste(labels, bins, sep=' ')

d <- read_feather(path(root, '../correlations_long.feather'))
d <- d %>% 
    filter(!network %in% c('dATN', 'DMN'), lag <= 10) 

# Standard correlation analysis

cd <- d %>% 
    mutate(band = cut(frequency, breaks, labels)) %>% 
    filter(band != 'init [0, 1)') %>% 
    group_by(subject, session, channel, band, lag, network) %>% 
    summarize(cors = mean(cors)) %>% 
    group_by(subject, band, lag, network) %>% 
    summarize(cors = mean(cors)) %>% 
    group_by(band, lag, network) %>% 
    summarize(cors_ = mean(cors), p  = t.test(cors, mu = 0)$p.value) %>% 
    group_by(network) %>% 
    mutate(p_adj = p.adjust(p, method='fdr')) %>% 
    ungroup() %>% 
    filter(p_adj < .05) 

r_range <- range(abs(cd$cors_))
pwr.r.test(r=r_range[1], power=.8)

# Zoom in on the silly low correlation

td <- d %>% 
    mutate(band = cut(frequency, breaks, labels)) %>% 
    filter(band != 'init [0, 1)') %>% 
    group_by(subject, session, channel, band, lag, network) %>% 
    summarize(cors = mean(cors)) %>% 
    group_by(subject, band, lag, network) %>% 
    summarize(cors = mean(cors)) %>% 
    filter(band == 'Gamma (30,40]', lag == 10, network == 'DNb')

t_test <- t.test(td$cors, mu=0)
t <- round(t_test$statistic, 3)
df <- t_test$parameter
p <- round(t_test$p.value, 3)
p_adj <- round(cd[abs(cd$cors_) == min(abs(cd$cors_)),]$p_adj, 3)
m <- round(mean(td$cors),6)
N <- length(unique(td$subject))
se  <- round(sd(td$cors) / sqrt(N), 6)

label <- glue('Mean = {m}\nSE = {se}\nt({df}) = {t}\np = {p}\np (FDR) = {p_adj}')
ct <- td[1,]
caption <- glue('Band: {ct$band}, lag: {ct$lag}, network: {ct$network}')

td %>% 
    ggplot(aes(x = cors)) + 
    geom_vline(xintercept = 0, linetype = 'dashed') + 
    geom_histogram(fill = 'steelblue', color = 'black', bins = 15, alpha = .7) +
    annotate('text', label=label, x = -.05, y = 3, size = 7, hjust=0) + 
    xlim(-.05, .05) + 
    labs(
        x = 'Average Spearman correlation',
        y = 'Frequency',
        caption = caption
    ) + 
    theme_bw() + 
    theme(axis.ticks = element_blank(),
          panel.grid = element_blank(),
          text = element_text(size = 16))

ggsave(path(root, 'low_correlation_inspection.png'), 
       height = 8, width = 10, units = 'in', dpi = 300)

# -- LOW -- #
# approximate correlation power calculation (arctangh transformation) 
# 
# n = 122708
# r = 0.007997631
# sig.level = 0.05
# power = 0.8
# alternative = two.sided

# -- HIGH -- #
# approximate correlation power calculation (arctangh transformation) 
# 
# n = 490.6689
# r = 0.1260742
# sig.level = 0.05
# power = 0.8
# alternative = two.sided


# ICC analysis
get_icc <- function(s1, s2) {
    d <- data.frame(s1, s2)
    i <- psych::ICC(d)$results
    r <- i[i$type=='ICC3',]$ICC
    p <- i[i$type=='ICC3',]$p
    return(tibble(r=r, p=p))
}

cd <- d %>% 
    filter(subject != 'sub-023') %>% 
    group_by(subject, session, frequency, lag, network) %>% 
    summarize(cors = mean(cors)) %>% 
    ungroup() %>% 
    mutate(session = str_replace(session, '-', '')) %>% 
    spread(session, cors) %>% 
    group_by(frequency, lag, network) %>% 
    summarize(get_icc(ses001, ses002),
              .groups = 'drop') %>% 
    group_by(network) %>% 
    mutate(p_adj = p.adjust(p, method='fdr')) %>% 
    ungroup() %>% 
    filter(p_adj < .05)

r_range <- range(abs(cd$r))
pwr.r.test(r=r_range[1], power=.8)

# approximate correlation power calculation (arctangh transformation) 
# 
# n = 52.47483
# r = 0.3756776
# sig.level = 0.05
# power = 0.8
# alternative = two.sided

pwr.r.test(r=r_range[2], power=.8)

# approximate correlation power calculation (arctangh transformation) 
# 
# n = 6.626541
# r = 0.8842785
# sig.level = 0.05
# power = 0.8
# alternative = two.sided



















