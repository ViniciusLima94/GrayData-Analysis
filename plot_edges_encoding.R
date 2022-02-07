# Libraries
library(ggraph)
library(igraph)
library(tidyverse)
library(RColorBrewer)

################################################################################
# Loading data
################################################################################
root = "/home/vinicius/funcog/gda"

df_coh = read.csv(
  paste(
    c(root,
      "/Results/lucy/mutual_information/mi_coh.csv"),
    collapse="")
)

df_plv = read.csv(
  paste(
    c(root,
      "/Results/lucy/mutual_information/mi_plv.csv"),
    collapse="")
)

df <- df_coh %>% select(1:5)
df$plv <- df_plv$plv
df$t <- NULL
df$s <- NULL

################################################################################
# Number of sig. effects
################################################################################
freqs <- unique(df$freqs)
times <- unique(df$times)

n_freqs <- length(freqs)
n_times <- length(times)

neff <- data.frame(matrix(ncol = 4, nrow = 0))
x <- c("freqs", "times", "metric", "n")
colnames(neff) <- x

for(f in freqs) {
  for(t in times) {
    # Filter for a freq and time
    out <- df %>% filter(freqs==f,
                         times==t)
    # Coherence
    neff_coh <- sum(out$coh > 0)
    # PLV
    neff_plv <- sum(out$plv > 0)
    # Row for dataframe
    row <- c(f, t, "coh", neff_coh)
    neff[nrow(neff) + 1,] <- row
    row <- c(f, t, "plv", neff_plv)
    neff[nrow(neff) + 1,] <- row
  }
}

# Define labels
freqs.labs <- c("3 Hz", "11 Hz", "19 Hz", "27 Hz", "35 Hz",
                "43 Hz", "51 Hz", "59 Hz", "67 Hz", "75 Hz")
names(freqs.labs) <- list(3, 11, 19, 27, 35, 43, 51, 59, 67, 75)

times.labs <- c("baseline", "cue", "e. delay", "l. delay", "match")
names(times.labs) <- 0:4

neff %>% ggplot(aes(x=times, y = n, group=metric)) +
  geom_line(aes(color=metric))  + 
  geom_point(aes(color=metric)) + 
  facet_wrap(~freqs, ncol=5,
             labeller = labeller(freqs = freqs.labs)) +
  scale_x_discrete(labels=times.labs) +
  theme_classic() +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.x = element_text(angle = 45, hjust=1)) +
  labs(x = "", y = "#sig. effects")