library(tidyverse)
library(ggraph)
library(tidygraph)
library(readxl)
library(glue)
library(shadowtext)


raw_dat <- read_excel("data/data_action.xlsx") %>%
  mutate(across(everything(), as.character)) %>%
  replace(is.na(.), "Unknown")

graph_data <- raw_dat %>%
  mutate(
    root_id = "Origin",
    
    # 1. Target
    target_id = target,
    
    # 2. Category
    category_id = paste(target, category, sep = "__"),
    
    # 3. Activity
    activity_id = paste(target, category, activity, sep = "__"), 
    
    # 4. Name
    name_label = ifelse(name == activity, paste0(name, " "), name),
    name_id = name 
  )


edges <- bind_rows(
  graph_data %>% distinct(from = root_id,     to = target_id),
  graph_data %>% distinct(from = target_id,   to = category_id),
  graph_data %>% distinct(from = category_id, to = activity_id),
  graph_data %>% distinct(from = activity_id, to = name_id)
)


nodes <- bind_rows(
  data.frame(name = "Origin", type = "Root", label = "Origin"),
  
  # Target 
  graph_data %>% distinct(name = target_id, label = target) %>% mutate(type = "Target"),
  
  # Category 
  graph_data %>% distinct(name = category_id, label = category) %>% mutate(type = "Category"),
  
  # Activity
  graph_data %>% distinct(name = activity_id, label = activity) %>% mutate(type = "Activity"),
  
  # Name
  graph_data %>% distinct(name = name_id, label = name_label) %>% mutate(type = "Name")
)


graph_obj <- tbl_graph(nodes = nodes, edges = edges)
layout_data <- create_layout(graph_obj, layout = 'dendrogram', circular = FALSE)

cols <- c(
  "Target" = "#0530AD",
  "Category" = "#E50914",
  "Activity" = "#228B22",
  "Name" = "#333333"
)

gg <- ggraph(layout_data) +
  geom_edge_diagonal(color = '#999999', linewidth = 0.2, alpha = 0.6) +

  # 1. Name
  shadowtext::geom_shadowtext(
    data = layout_data %>% filter(type == "Name"),
    aes(x, y, label = label, color = type), 
    size = 3,
    angle = 90, 
    hjust = 0,
    nudge_y = 0.05,
    fontface = "plain",
    bg.color = '#FFFFFF',
    bg.r = 0.1
  ) +
  
  # 2. Activity
  shadowtext::geom_shadowtext(
    data = layout_data %>% filter(type == "Activity"),
    aes(x, y, label = str_to_title(label), color = type),
    size = 4,
    angle = 90,
    hjust = 0.5,
    nudge_y = 0,      
    fontface = "bold",
    bg.color = '#FFFFFF'
  ) +
  
  # 3. Category
  shadowtext::geom_shadowtext(
    data = layout_data %>% filter(type == "Category"),
    aes(x, y, label = str_to_title(label), color = type),
    size = 6,
    angle = 90,
    hjust = 1,
    nudge_y = -0.05,    
    fontface = "bold",
    bg.color = '#FFFFFF'
  ) +
  
  # 4. Target
  geom_text(
    data = layout_data %>% filter(type == "Target"),
    aes(x, y, label = str_wrap(str_to_title(label), 10), color = type),
    size = 8.5,
    angle = 90,
    hjust = 1,
    nudge_y = -0.2,
    fontface = "bold",
    lineheight = 0.8
  ) +

  scale_color_manual(values = cols) +
  guides(color = 'none') +
  
  scale_x_continuous(expand = expansion(add = c(0.5, 0.5))) +
  scale_y_reverse() + 
  coord_radial(
    rotate.angle = TRUE,
    inner.radius = 0.01,
    start = 0 * pi,
    end = 2 * pi,
    clip = "off"
  ) +
  
  #labs(
  #  title = "Hierarchy of Actions",
  #  subtitle = paste("Total Actions:", sum(layout_data$type == "Name"))
  #) +
  
  theme_void() +
  theme(
    plot.background = element_rect(fill = '#FFFFFF', color = NA),
    plot.margin = margin(-80, -80, -80, -80),
    plot.title = element_text(hjust = 0.5, face = "bold", size = 20, margin = margin(120, 0, -100, 0)),
    plot.subtitle = element_text(hjust = 0.5, size = 12, color = "grey50", margin = margin(100, 0, -80, 0))
  )

wkdir <- getwd()
width = 18
height = 18
ggsave(gg, filename = glue('{wkdir}/figures/action_hierarchy_final.pdf'),
       width = width, height = height, limitsize = FALSE)