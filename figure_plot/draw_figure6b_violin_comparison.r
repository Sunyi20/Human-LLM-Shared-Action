if(!requireNamespace("ggpubr", quietly = TRUE)) {
  stop("Package 'ggpubr' is required. Please install it once with install.packages('ggpubr').")
}
library(ggplot2)
library(dplyr)
library(ggpubr) # 专门用于添加显著性星号和出版级绘图

# 基础路径
base_path <- "data/pycortex"

# 读取4个模型的数据文件
# Voxel encoding
LLM_deepseek_voxel <- read.csv(file.path(base_path, "voxel_encoding_results_csv/LLM_deepseek_voxel_encoding_results.csv"))
LLM_qwen7B_voxel <- read.csv(file.path(base_path, "voxel_encoding_results_csv/LLM_qwen_7B_voxel_encoding_results.csv"))
MLLM_qwen7B_voxel <- read.csv(file.path(base_path, "voxel_encoding_results_csv/MLLM_qwen_7B_voxel_encoding_results.csv"))
MLLM_qwen72B_voxel <- read.csv(file.path(base_path, "voxel_encoding_results_csv/MLLM_qwen_72B_voxel_encoding_results.csv"))

# Exclude subject 17 for Voxel encoding
LLM_deepseek_voxel <- LLM_deepseek_voxel[-17, ]
LLM_qwen7B_voxel <- LLM_qwen7B_voxel[-17, ]
MLLM_qwen7B_voxel <- MLLM_qwen7B_voxel[-17, ]
MLLM_qwen72B_voxel <- MLLM_qwen72B_voxel[-17, ]

# Searchlight
LLM_deepseek_searchlight <- read.csv(file.path(base_path, "searchlight_results_csv/LLM_deepseek_searchlight_results.csv"))
LLM_qwen7B_searchlight <- read.csv(file.path(base_path, "searchlight_results_csv/LLM_qwen_7B_searchlight_results.csv"))
MLLM_qwen7B_searchlight <- read.csv(file.path(base_path, "searchlight_results_csv/MLLM_qwen_7B_searchlight_results.csv"))
MLLM_qwen72B_searchlight <- read.csv(file.path(base_path, "searchlight_results_csv/MLLM_qwen_72B_searchlight_results.csv"))

# 将单个模型的数据转换为长格式
reshape_model_data <- function(model_data, model_name, method_name) {
  data_list <- list()
  for(i in 1:nrow(model_data)) {
    for(j in 2:ncol(model_data)) {
      data_list[[length(data_list) + 1]] <- data.frame(
        model_type = model_name,
        value = model_data[i, j],
        region = names(model_data)[j],
        method = method_name
      )
    }
  }
  do.call(rbind, data_list)
}

# 创建voxel encoding数据（4个模型）
voxel_data <- rbind(
  reshape_model_data(LLM_deepseek_voxel, "Deepseek-R1", "Voxel_Encoding"),
  reshape_model_data(LLM_qwen7B_voxel, "Qwen2.5-7B", "Voxel_Encoding"),
  reshape_model_data(MLLM_qwen7B_voxel, "Qwen2.5-VL-7B", "Voxel_Encoding"),
  reshape_model_data(MLLM_qwen72B_voxel, "Qwen2.5-VL-72B", "Voxel_Encoding")
)

# 创建searchlight数据（4个模型）
searchlight_data <- rbind(
  reshape_model_data(LLM_deepseek_searchlight, "Deepseek-R1", "Searchlight"),
  reshape_model_data(LLM_qwen7B_searchlight, "Qwen2.5-7B", "Searchlight"),
  reshape_model_data(MLLM_qwen7B_searchlight, "Qwen2.5-VL-7B", "Searchlight"),
  reshape_model_data(MLLM_qwen72B_searchlight, "Qwen2.5-VL-72B", "Searchlight")
)

# 设定模型顺序
model_levels <- c("Deepseek-R1", "Qwen2.5-7B", "Qwen2.5-VL-7B", "Qwen2.5-VL-72B")
voxel_data$model_type <- factor(voxel_data$model_type, levels = model_levels)
searchlight_data$model_type <- factor(searchlight_data$model_type, levels = model_levels)

# 定义需要比较的模型对（有意义的对比）
comparisons <- list(
  c("Deepseek-R1", "Qwen2.5-7B"),      # LLM之间比较
  c("Qwen2.5-VL-7B", "Qwen2.5-VL-72B"),    # MLLM之间比较（规模效应）
  c("Qwen2.5-7B", "Qwen2.5-VL-7B"),      # 同规模LLM vs MLLM
  c("Deepseek-R1", "Qwen2.5-VL-72B")     # 最优LLM vs 最优MLLM
)

# 4色配色方案
model_colors <- c(
  "Deepseek-R1"    = "#334D66",
  "Qwen2.5-7B"     = "#3F8F68",
  "Qwen2.5-VL-7B"  = "#D89A2B",
  "Qwen2.5-VL-72B" = "#8E4A6D"
)

# 绘制4模型对比图
draw_comparison_plot <- function(data, panel_label) {
  regions <- unique(as.character(data$region))
  n_models <- length(model_levels)
  dodge_width <- 0.82
  global_max <- max(data$value, na.rm = TRUE)
  global_min <- min(data$value, na.rm = TRUE)
  value_span <- global_max - global_min
  if(isTRUE(all.equal(value_span, 0))) value_span <- max(abs(global_max), 0.01)

  data$region <- factor(data$region, levels = regions)

  # 模型在dodge内的偏移位置（4个模型时）
  offsets <- seq(-dodge_width/2 + dodge_width/(2*n_models),
                 dodge_width/2 - dodge_width/(2*n_models),
                 length.out = n_models)
  names(offsets) <- model_levels

  # 仅保留显著比较，避免在图中出现 ns
  sig_annotations <- list()
  for(region in regions) {
    region_data <- data[data$region == region, ]
    x_pos <- which(regions == region)
    sig_index <- 0

    for(pair in comparisons) {
      vals_a <- region_data[region_data$model_type == pair[1], "value"]
      vals_b <- region_data[region_data$model_type == pair[2], "value"]

      if(length(vals_a) < 2 || length(vals_b) < 2) next

      p_value <- t.test(vals_a, vals_b)$p.value
      if(is.na(p_value) || p_value >= 0.05) next

      sig_index <- sig_index + 1
      sig_label <- ifelse(p_value < 0.001, "***",
                          ifelse(p_value < 0.01, "**", "*"))

      sig_annotations[[length(sig_annotations) + 1]] <- data.frame(
        region = region,
        x_left = x_pos + offsets[pair[1]],
        x_right = x_pos + offsets[pair[2]],
        x_mid = x_pos + mean(offsets[pair]),
        y = global_max + value_span * (0.08 + 0.07 * (sig_index - 1)),
        label = sig_label
      )
    }
  }

  annotation_df <- if(length(sig_annotations) > 0) bind_rows(sig_annotations) else NULL
  y_top <- if(is.null(annotation_df)) {
    global_max + value_span * 0.10
  } else {
    max(annotation_df$y) + value_span * 0.08
  }

  p <- ggplot(data, aes(x = region, y = value, fill = model_type)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray65", linewidth = 0.28) +
    geom_violin(position = position_dodge(width = dodge_width),
                trim = FALSE,
                alpha = 0.42,
                color = NA,
                scale = "width",
                linewidth = 0.2) +
    geom_point(
      aes(color = model_type),
      position = position_jitterdodge(
        jitter.width = 0.08,
        dodge.width = dodge_width,
        seed = 123
      ),
      size = 0.75,
      alpha = 0.42,
      stroke = 0
    ) +
    geom_boxplot(position = position_dodge(width = dodge_width),
                 width = 0.09,
                 outlier.shape = NA,
                 alpha = 0.78,
                 linewidth = 0.28,
                 color = "gray25") +
    stat_summary(
      aes(group = model_type),
      fun = median,
      geom = "point",
      position = position_dodge(width = dodge_width),
      shape = 21,
      size = 1.5,
      stroke = 0.25,
      fill = "white",
      color = "gray15"
    ) +
    scale_fill_manual(values = model_colors) +
    scale_color_manual(values = model_colors, guide = "none") +
    labs(tag = panel_label, y = "Decoding Accuracy", x = "Brain Region", fill = "Model") +
    coord_cartesian(ylim = c(global_min - value_span * 0.05, y_top), clip = "off") +
    theme_classic(base_size = 13) +
    theme(
      plot.tag = element_text(size = 14, face = "bold", hjust = 0, vjust = 1),
      plot.tag.position = c(0.01, 0.99),
      legend.position = "top",
      legend.title = element_blank(),
      legend.key.size = grid::unit(0.55, "cm"),
      legend.text = element_text(size = 18),
      axis.title.x = element_text(size = 20, color = "black"),
      axis.title.y = element_text(size = 20, color = "black"),
      axis.line.x = element_line(linewidth = 0.4, color = "black"),
      axis.line.y = element_line(linewidth = 0.4, color = "black"),
      axis.ticks = element_line(linewidth = 0.3, color = "black"),
      axis.text.x = element_text(size = 18, angle = 35, hjust = 1, vjust = 1, color = "black"),
      axis.text.y = element_text(size = 18, color = "black"),
      panel.grid = element_blank(),
      plot.margin = margin(8, 18, 6, 6)
    )

  if(!is.null(annotation_df)) {
    bracket_height <- value_span * 0.025
    p <- p +
      geom_segment(
        data = annotation_df,
        aes(x = x_left, xend = x_right, y = y, yend = y),
        inherit.aes = FALSE,
        linewidth = 0.35
      ) +
      geom_segment(
        data = annotation_df,
        aes(x = x_left, xend = x_left, y = y - bracket_height, yend = y),
        inherit.aes = FALSE,
        linewidth = 0.35
      ) +
      geom_segment(
        data = annotation_df,
        aes(x = x_right, xend = x_right, y = y - bracket_height, yend = y),
        inherit.aes = FALSE,
        linewidth = 0.35
      ) +
      geom_text(
        data = annotation_df,
        aes(x = x_mid, y = y + value_span * 0.02, label = label),
        inherit.aes = FALSE,
        size = 3.2,
        vjust = 0,
        fontface = "bold"
      )
  }

  return(p)
}

# 绘制Voxel Encoding图
voxel_plot <- draw_comparison_plot(voxel_data, "A")

# 绘制Searchlight图
searchlight_plot <- draw_comparison_plot(searchlight_data, "B")

# 保存图片到脚本同目录下
output_path_voxel <- paste0(getwd(), "/voxel_encoding_comparison.pdf")
output_path_searchlight <- paste0(getwd(), "/searchlight_comparison.pdf")

ggsave(output_path_voxel, voxel_plot, width = 25, height = 7)
ggsave(output_path_searchlight, searchlight_plot, width = 25, height = 7)

# 显示图片
print(voxel_plot)
print(searchlight_plot)
