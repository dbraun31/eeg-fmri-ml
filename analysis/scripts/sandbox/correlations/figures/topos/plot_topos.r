plot_dna_through_dan_topo <- function() {

	# Get frequency bands
	breaks <- c(0, 1, 4, 8, 12, 30, 40)
	labels <- c('init', 'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma')
	bins <- unique(cut(result$frequency, breaks=breaks))
	labels <- paste(labels, bins, sep=' ')

	# Prep data
	pd1 <- result %>% 
	    filter(network %in% c('DNa', 'DNb', 'DAN')) %>%
	    group_by(subject, lag, frequency, network) %>% 
	    summarize(cors = mean(cors)) %>% 
	    group_by(lag, frequency, network) %>% 
	    summarize(cors = mean(cors)) 
	
	pd2 <- result %>% 
		mutate(band = cut(frequency, breaks, labels)) %>% 
		filter(band != 'init (0,1]') %>% 
		inner_join(py$ch_pos) %>% 
		group_by(subject, x, y, band, network) %>% 
		summarize(cors = mean(cors), channel = unique(channel)) %>% 
		group_by(x, y, band, network) %>% 
		summarize(cors = mean(cors), channel = unique(channel)) %>% 
		mutate(z = 50, 
			   network = recode(network, `dan` = 'DAN', `dmn` = 'DMN',
							   `DNa` = 'DNa', `DNb` = 'DNb'))

	small <- floor(min(pd1$cors, pd2$cors)*100)/100
	big <- ceiling(max(pd1$cors, pd2$cors)*100)/100

	p2 <- pd2 %>% 
		filter(band == 'Alpha (8,12]',
			   network %in% c('DNa', 'DNb', 'DAN')) %>%
		mutate(network = factor(network, levels = c('DNa', 'DNb', 'DAN'))) %>%
		ggplot(aes(x = x, y = y, z = z)) + 
		geom_topo(chan_markers = 'text', aes(fill = cors, label = channel)) +
		facet_wrap(~network) + 
		scale_fill_gradientn(colors = rev(brewer.pal(11, 'RdBu')),
							 values = rescale(c(small, 0, big)),
							 breaks = c(small, 0, big),
							 limits = c(small, big)) +
		labs(
			x = '',
			y = '',
			fill = latex2exp::TeX('$\\rho_{EEG, fMRI}$')
		) + 
		theme_bw() + 
		theme(panel.grid = element_blank(),
			  axis.text = element_blank(),
			  axis.title = element_blank(),
			  strip.background = element_rect(fill = NA),
			  legend.position = 'none',
			  text = element_text(size=size),
			  axis.ticks = element_blank())

	return(p2)

}
