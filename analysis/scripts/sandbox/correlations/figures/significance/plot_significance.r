# -- FREQ X LAG WITH SIGNIFICANCE -- #

plot_significance <- function(networks=c(), label_middle=TRUE) {
	breaks <- c(0, 1, 4, 8, 12, 30, 40)
	labels <- c('init', 'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma')
	bins <- unique(cut(result$frequency, breaks=breaks))
	labels <- paste(labels, bins, sep=' ')
	
	if (length(networks) == 0) {
    	networks <- c('DNa', 'DNb', 'DANa', 'DANb', 'FPNa', 'FPNb', 'SAL')
	}

	ps <- result %>% 
		mutate(bin = cut(frequency, breaks, labels)) %>% 
		filter(bin != 'init (0,1]', lag <= 10,
			   network %in% !!networks) %>% 
		group_by(subject, lag, network, bin) %>% 
		summarize(cors = mean(cors)) %>% 
		group_by(lag, network, bin) %>% 
		summarize(p = t.test(cors, mu = 0)$p.value) %>% 
	    group_by(network) %>% 
		mutate(p_adj = p.adjust(p, method='fdr'),
			   network = factor(network, levels = !!networks)) %>% 
		filter(p_adj < .05) 
		

	pd <- result %>% 
		mutate(bin = cut(frequency, breaks, labels)) %>% 
		filter(bin != 'init (0,1]', lag <= 10,
			   network %in% !!networks) %>% 
		group_by(bin, lag, network) %>% 
		summarize(cors = mean(cors)) %>% 
		mutate(network = factor(network, levels = networks))
    if (label_middle) {
		sfg <- scale_fill_gradientn(colors = rev(brewer.pal(11, 'RdBu')),
							 values = rescale(c(small, 0, big)),
							 breaks = c(small, 0, big),
							 limits = c(small, big))  
    } else {
		sfg <- scale_fill_gradientn(colors = rev(brewer.pal(11, 'RdBu')),
							 values = rescale(c(small, 0, big)),
							 breaks = c(small, big),
							 limits = c(small, big))  
    }

	# Enlarged                         
	p <- pd %>%     
		ggplot(aes(x = bin, y = lag)) + 
		geom_tile(aes(fill = cors)) + 
		geom_point(data=ps, aes(x = bin, y = lag), shape = 8, color = 'gold', size = 3) + 
		facet_wrap(~network, nrow=1) + 
		scale_y_continuous(breaks = seq(0, 10, 2), labels = seq(0, 10, 2)) +
		labs(
			x = 'Frequency bin',
			y = 'Lag (s)',
			fill = latex2exp::TeX('$\\rho_{~~EEG, fMRI}$')
		) + 
	    sfg + 
		theme_bw() + 
		theme(strip.background = element_rect(fill = NA),
			  axis.ticks = element_blank(),
			  panel.grid = element_blank(),
			  legend.position = 'bottom',
			  text = element_text(size = 25),
			  axis.text.x = element_text(angle = 45, hjust=1, size = 14),
			  legend.text = element_text(size = 16, angle = 45, hjust=1),
			  legend.title = element_text(margin = margin(r = 30)))
		
	return (p)
}
