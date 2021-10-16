library(ggplot2)
library(magrittr)
library(pheatmap)
library(viridis)
setwd("~/Documents/beehive/multi-group-GP/experiments/gtex/")


a_mat <- read.csv("./a_matrix_full.csv",
                  row.names = 1,
                  header = T)
rownames(a_mat) <- colnames(a_mat)
a_mat <- log10(a_mat)
diag(a_mat) <- NA

paletteLength <- 50
myColor <-
  colorRampPalette(c("red", "white", "black"))(paletteLength)

tissue_names <- colnames(a_mat)
tissue_names <- stringr::str_replace_all(tissue_names, "[.]", " ")

quantile_breaks <- function(xs, n = 10) {
  breaks <- quantile(xs, probs = seq(0, 1, length.out = n))
  breaks[!duplicated(breaks)]
}

mat_breaks <- quantile_breaks(a_mat %>% as.matrix(), n = 11)


pheatmap::pheatmap(
  a_mat,
  main = "Estimated pairwise log(a)",
  color = myColor,
  # fontsize = 15,
  fontsize = 30,
  filename = "../../plots/gtex_pairwise_a_heatmap.png",
  width=16,
  height=16,
  show_colnames = T,
  labels_row = tissue_names,
  labels_col = tissue_names,
  # color=inferno(length(mat_breaks) - 1, direction = -1),
  # breaks = mat_breaks
)
