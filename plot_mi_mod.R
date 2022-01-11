library(tidyverse)
library(umap)


# Titles for frequencies
freqs.labs <- c("3 Hz", "11 Hz", "19 Hz", "27 Hz", "35 Hz",
                "43 Hz", "51 Hz", "59 Hz", "67 Hz", "75 Hz")
stages <- c("baseline", "cue", "early delay", "late delay", "match")
colors <- c("blue", "orange", "green", "red", "purple")
names(freqs.labs) <- list(3, 11, 19, 27, 35, 43, 51, 59, 67, 75)

# Stages names
stages = c("Baseline", "Cue", "Early delay", "Late delay", "Match")

df = read.csv(
            "Results/lucy/mutual_information/mi_mod_df.csv"
)
df$roi <- NULL

ggplot(data=df, aes(x=freqs, y=mod, fill=as.factor(times))) +
geom_bar(alpha = 0.7, stat='identity', position="stack") +
labs(x="MI(avg. modularity, stimulus)", y="Freqs [Hz]", fill = "Stages") +
scale_fill_manual(labels = stages, values = colors) + 
ggtitle("MI(avg. modularity, stimulus)") +
scale_x_discrete(breaks=unique(df$freqs), label=freqs.labs) +
theme(plot.title = element_text(hjust=0.5))

ggsave(
paste(c("figures/mi_mod_0.png"), collapse = ""),
width = 14, height = 4)
