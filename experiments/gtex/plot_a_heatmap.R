library(ggplot2)
library(magrittr)
library(pheatmap)
setwd("~/Documents/beehive/multi-group-GP/experiments/gtex/")


a_mat <- read.csv("./a_matrix_full.csv",
                  row.names = 1,
                  header = T)
rownames(a_mat) <- colnames(a_mat)
a_mat <- log10(a_mat)
diag(a_mat) <- 0

paletteLength <- 50
myColor <-
  colorRampPalette(c("red", "white", "black"))(paletteLength)

tissue_names <- colnames(a_mat)
tissue_names <- stringr::str_replace_all(tissue_names, "[.]", " ")

pheatmap::pheatmap(
  a_mat,
  main = "Estimated pairwise log(a)",
  color = myColor,
  fontsize = 15,
  filename = "../../plots/gtex_pairwise_a_heatmap.png",
  width=16,
  height=16,
  show_colnames = T,
  labels_row = tissue_names,
  labels_col = tissue_names,
)
