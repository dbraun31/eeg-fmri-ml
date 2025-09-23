
plot_heat <- function(networks) {
	# Get frequency bands
	breaks <- c(0, 1, 4, 8, 12, 30, 40)
	labels <- c('init', 'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma')
	bins <- unique(cut(result$frequency, breaks=breaks))
	labels <- paste(labels, bins, sep=' ')

	# Prep data
	pd1 <- result %>% 
		filter(network %in% !!networks) %>%
		group_by(subject, lag, frequency, network) %>% 
		summarize(cors = mean(cors)) %>% 
		group_by(lag, frequency, network) %>% 
		summarize(cors = mean(cors)) 

	pd2 <- result %>% 
		mutate(band = cut(frequency, breaks, labels)) %>% 
		filter(band != 'init (0,1]') %>% 
		inner_join(ch_pos) %>% 
		group_by(subject, x, y, band, network) %>% 
		summarize(cors = mean(cors), channel = unique(channel)) %>% 
		group_by(x, y, band, network) %>% 
		summarize(cors = mean(cors), channel = unique(channel)) %>% 
		mutate(z = 50) 

	# Get frequency bands
	breaks <- c(0, 1, 4, 8, 12, 30, 40)
	labels <- c('init', 'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma')
	bins <- unique(cut(result$frequency, breaks=breaks))
	labels <- paste(labels, bins, sep=' ')

	small <- floor(min(pd1$cors, pd2$cors)*100)/100
	big <- ceiling(max(pd1$cors, pd2$cors)*100)/100

	# Plot
	p1 <- pd1 %>% 
		mutate(network = factor(network, levels = !!networks)) %>%
		ggplot(aes(x = frequency, y = lag)) +
		geom_tile(aes(fill = cors)) + 
		facet_wrap(~network) +
		scale_fill_gradientn(colors = rev(brewer.pal(11, 'RdBu')),
							 values = rescale(c(min(pd1$cors, pd2$cors), 0, 
												max(pd1$cors, pd2$cors))),
							 limits = c(small, big),
							 breaks = c(small, 0, big),
							 labels = c(small, 0, big)) + 
		labs(
			x = 'Frequency (Hz)',
			y = 'Lag (s)',
			fill = latex2exp::TeX('$\\rho_{~~EEG, fMRI}$')
		) + 
		scale_y_continuous(breaks = seq(0, max(result$lag), 2), labels = seq(0, max(result$lag), 2)) + 
		theme_bw() + 
		theme(strip.background = element_rect(fill = NA),
			  panel.grid = element_blank(),
			  axis.ticks = element_blank(),
			  axis.text = element_text(size = 18),
			  legend.position = 'bottom',
			  text = element_text(size=text))

	return(p1)

}
