library(tidyverse)
library(umap)

ROOT = "/home/vinicius/storage1/projects/GrayData-Analysis/figures"

# Titles for frequencies
freqs.labs <- c("3 Hz", "11 Hz", "19 Hz", "27 Hz", "35 Hz",
                "43 Hz", "51 Hz", "59 Hz", "67 Hz", "75 Hz")
names(freqs.labs) <- list(3, 11, 19, 27, 35, 43, 51, 59, 67, 75)

# Stages names
stages = c("Baseline", "Cue", "Early delay", "Late delay", "Match")

data_path = "/home/vinicius/funcog/gda/Results/lucy/mutual_information/"
file = paste(
  c(data_path, "mi_df_degree_fdr.csv"),
  collapse = "")

for(i in 0:4){
  df = read.csv(
      file
  )
  
  df <- df %>% gather(key = metric,
                      value = t_values,
                      c("power", "coh", "plv", "pec"))

  df <- df %>% filter(times==i)
  
  ggplot(data=df, aes(x=roi, y=t_values, fill = metric)) +
    geom_bar(alpha = 0.7, stat='identity', position="stack") +
    ylim(0, 80) +
    coord_flip() +
    facet_wrap(~freqs, ncol=5,
               labeller = labeller(freqs = freqs.labs)) +
    ggtitle(stages[i+1]) +
    theme(plot.title = element_text(hjust=0.5)) +
    labs(y="t-values", x="ROI")
  
  ggsave(
    paste(
      c(ROOT,
        "/mi_hist_degree_",
        i,
        ".png"),
      collapse = ""),
    width = 12, height = 8)
}

metrics <- c("coh", "plv", "pec")

for(metric in metrics) {
  # For each roi determines if the stim is encoded by power uniquely, by FC degree
  # uniquely or both.
  df = read.csv(
    file
  )
  
  out <- df %>% select(1:3)
  
  out$power <- as.integer(df$power > 0)
  if(metric == "coh") {
    out$fc <- 2*(df$coh > 0)
  }
  else if(metric == "plv") {
    out$fc <- 2*(df$plv > 0)
  }
  else {
    out$fc <- 2*(df$pec > 0)
  }
  out$n <- out$power + out$fc
  
  mycolors = c("#FFFFFF", "#D800FF", "#178A00", "#000000")
  times.labs <- c("baseline", "cue", "e. delay", "l. delay", "match")
  names(times.labs) <- 0:4
  
  out %>% ggplot(aes(x=factor(times), y=freqs, fill= as.factor(n))) + 
    scale_x_discrete(labels=times.labs) +
    geom_tile() +
    facet_wrap(~roi, ncol=8) +
    scale_fill_manual(values = mycolors, name="Encoding",
                      labels=c("No encoding","Power uniquely",
                               "Degree uniquely", "Both")) +
    theme_classic()  +
    theme(plot.title = element_text(hjust=0.5),
          axis.text.x = element_text(angle = 90, hjust=1)) +
    labs(x = "", y = "Freqs [Hz]") +
    ggtitle(paste(c(metric, " encoding"), collapse=""))
    #
  
  ggsave(
    paste(
      c(ROOT,
        "/mi_enconding_degree_",
        metric,
        ".png"),
      collapse = ""),
    width = 10, height = 8)
}
