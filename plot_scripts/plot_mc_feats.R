# Libraries
library(ggraph)
library(ggpubr)
library(igraph)
library(tidyverse)
library(RColorBrewer)

################################################################################
# Loading data
################################################################################
root = "/home/vinicius/funcog/gda"

results = "/home/vinicius/storage1/projects/GrayData-Analysis/figures/"

aMC = read.csv(
  paste(
    c(root,
      "/Results/lucy/meta_conn/aMC.csv"),
    collapse="")
)

Q = read.csv(
  paste(
    c(root,
      "/Results/lucy/meta_conn/Q.csv"),
    collapse="")
)

nu = read.csv(
  paste(
    c(root,
      "/Results/lucy/meta_conn/nu.csv"),
    collapse="")
)

aMC$X <- NULL
Q$X <- NULL
nu$X <- NULL

data_summary <- function(x) {
  m <- median(x)
  ymin <- m-sd(x)
  ymax <- m+sd(x)
  return(c(y=m,ymin=ymin,ymax=ymax))
}

################################################################################
# Plotting features
################################################################################
# Time labels
times.labs <- c("baseline", "cue", "e. delay", "l. delay", "match")
names(times.labs) <- 0:4
# Frequency labels
freqs.labs <- c("3 Hz", "11 Hz", "19 Hz", "27 Hz", "35 Hz",
                "43 Hz", "51 Hz", "59 Hz", "67 Hz", "75 Hz")
names(freqs.labs) <- c(3, 11, 19, 27, 35,
                       43, 51, 59, 67, 75)



################################################################################
# Average meta-connectivity
################################################################################
p1 <- aMC %>% ggplot(aes(x=as.factor(times), y=MC)) + 
  geom_boxplot(aes(color=as.factor(times), fill=as.factor(times)),
              alpha=.6,
              show.legend = F) +
  facet_wrap(~freqs, ncol=5, scales = "free_y",
             labeller = labeller(freqs = freqs.labs)) +
  #stat_summary(fun.data=data_summary, 
  #             geom="pointrange",
  #             color="black") +
  scale_x_discrete(labels=times.labs) +
  theme_minimal() + 
  theme(plot.title = element_text(hjust=0.5),
        axis.text.x = element_text(angle = 45, hjust=1)) +
  labs(x = "", y="Avg. meta-connectivity")

#ggsave(
#  paste(
#    c(results,
#      "/avg_mc.png"),
#    collapse = ""),
#  width = 10, height = 4)
 
################################################################################
# Viscosity
################################################################################
p2 <- nu %>% ggplot(aes(x=as.factor(times), y=nu)) + 
  geom_boxplot(aes(color=as.factor(times), fill=as.factor(times)),
               alpha=.6,
               show.legend = F) +
  facet_wrap(~freqs, ncol=5, scales = "free_y",
             labeller = labeller(freqs = freqs.labs)) +
  #stat_summary(fun.data=data_summary, 
  #             geom="pointrange",
  #             color="black") +
  scale_x_discrete(labels=times.labs) +
  theme_minimal() + 
  theme(plot.title = element_text(hjust=0.5),
        axis.text.x = element_text(angle = 45, hjust=1)) +
  labs(x = "", y="Viscosity")

#ggsave(
#  paste(
#    c(results,
#      "/nu_mc.png"),
#    collapse = ""),
#  width = 10, height = 4)

################################################################################
# Modularity
################################################################################
p3 <- Q %>% ggplot(aes(x=as.factor(times), y=Q)) + 
  geom_boxplot(aes(color=as.factor(times), fill=as.factor(times)),
               alpha=.6,
               show.legend = F) +
  facet_wrap(~freqs, ncol=5, scales = "free_y",
             labeller = labeller(freqs = freqs.labs)) +
  #stat_summary(fun.data=data_summary, 
  #             geom="pointrange",
  #             color="black") +
  scale_x_discrete(labels=times.labs) +
  theme_minimal() + 
  theme(plot.title = element_text(hjust=0.5),
        axis.text.x = element_text(angle = 45, hjust=1)) +
  labs(x = "", y="Modularity")

#ggsave(
#  paste(
#    c(results,
#      "/Q_mc.png"),
#    collapse = ""),
#  width = 10, height = 4)

ggarrange(p1, p2, p3,
          ncol = 1, nrow = 3,
          labels = c("A", "B", "C")) 

ggsave(
  paste(
    c(results,
      "/mc_features.png"),
    collapse = ""),
  width = 10, height = 4)