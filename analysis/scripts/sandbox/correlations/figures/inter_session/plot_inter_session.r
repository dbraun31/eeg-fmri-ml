plot_inter_session <- function(network, label_middle=TRUE) {
	get_icc <- function(s1, s2) {
		d <- data.frame(s1, s2)
		i <- ICC(d)$results
		out <- i[i$type=='ICC3',]$ICC
		return(out)
	}

	pd <- result_run %>%
		filter(network == !!network, subject != 'sub-023', lag <= 10) %>%
		group_by(subject, session, frequency, lag) %>%
		summarize(cors = mean(cors, na.rm=TRUE)) %>%
		mutate(session = str_replace(session, '-', '')) %>%
		spread(session, cors) %>%
		group_by(frequency, lag) %>%
		summarize(icc = get_icc(ses001, ses002)) 

	pal <- paletteer::paletteer_c('ggthemes::Green', n = 100)

	p1 <- pd %>%
		ggplot(aes(x = frequency, y = lag)) +
		geom_tile(aes(fill = icc)) + 
		labs(
			 x = 'Frequency (Hz)',
			 y = 'Lag (s)',
			 main = 'Intraclass Correlation Coefficient',
			 fill = 'ICC') + 
		scale_fill_gradientn(colors = pal) + 
		theme_bw() + 
		theme(legend.position = 'bottom',
			  axis.ticks = element_blank(),
			  panel.grid = element_blank(),
			  legend.title = element_text(margin = margin(r = 10)),
			  text = element_text(size = text))

	print(glue::glue('Mean ICC: {round(mean(pd$icc), 2)}, SD = {round(sd(pd$icc), 2)}'))

	pd2 <- result_run %>%
		filter(network == !!network, lag <= 10) %>%
		group_by(subject, session, frequency, lag) %>%
		summarize(cors = mean(cors)) %>%
		group_by(session, frequency, lag) %>%
		summarize(cors = mean(cors)) 

	small <- floor(min(pd2$cors) * 100) / 100
	big <- ceiling(max(pd2$cors) * 100) / 100


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
	
	p2 <- pd2 %>%
		mutate(session = recode(session, `ses-001` = 'Session 1', `ses-002` = 'Session 2')) %>% 
		ggplot(aes(x = frequency, y = lag)) +
		geom_tile(aes(fill = cors)) + 
		labs(
			 x = 'Frequency (Hz)',
			 y = 'Lag (s)',
			 title = glue('{network} network'),
			 fill = latex2exp::TeX('$\\rho_{~~EEG,fMRI}$')) + 
	    sfg + 
		facet_wrap(~session) + 
		theme_bw() + 
		theme(legend.position = 'bottom',
			  axis.ticks = element_blank(),
			  panel.grid = element_blank(),
			  legend.text = element_text(angle = 55, hjust=1),
			  strip.background = element_rect(fill = NA),
			  text = element_text(size = text))
	
	# Return only averages
	return(p2)
		
}
