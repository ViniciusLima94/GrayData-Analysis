library(tidyverse)
library(umap)


# Titles for frequencies
freqs.labs <- c("3 Hz", "11 Hz", "19 Hz", "27 Hz", "35 Hz",
                "43 Hz", "51 Hz", "59 Hz", "67 Hz", "75 Hz")
names(freqs.labs) <- list(3, 11, 19, 27, 35, 43, 51, 59, 67, 75)

# Stages names
stages = c("Baseline", "Cue", "Early delay", "Late delay", "Match")

for(i in 0:4){
  df = read.csv(
                "Results/lucy/mutual_information/mi_df.csv"
  )
  
  df <- df %>% gather(key = metric, value = t_values, c("power", "degree", "coreness", "efficiency"))

  df <- df %>% filter(times==i)
  
  ggplot(data=df, aes(x=roi, y=t_values, fill = metric)) +
    geom_bar(alpha = 0.7, stat='identity', position="stack") +
    ylim(-10, 80) +
    coord_flip() +
    facet_wrap(~freqs, ncol=5,
               labeller = labeller(freqs = freqs.labs)) +
    ggtitle(stages[i+1]) +
    theme(plot.title = element_text(hjust=0.5))
  
  ggsave(
    paste(c("figures/mi_hist_", i, ".png"), collapse = ""),
    width = 12, height = 8)
}
