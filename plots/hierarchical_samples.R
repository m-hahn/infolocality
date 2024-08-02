data = read.csv("../results/hierarchical_orders_0.010853724928633568_0.09425584222966477_0.07298316323451004.txt", sep=",")
library(dplyr)
library(tidyr)
library(ggplot2)

data = data[order(data$ee),]
data$x = (1:nrow(data))

plot = ggplot(data, aes(x=x, y=ee, color=is_well_nested)) + geom_point()

library(dplyr)
library(tidyr)
library(ggplot2)
library(purrr)
library(stringr)

# Get all relevant file names
file_names <- list.files(path = "../results", 
                         pattern = "hierarchical_orders_.*\\.txt$", 
                         full.names = TRUE)

# Function to read a file and extract parameters from filename
read_file_with_params <- function(file_path) {
  # Extract parameters from filename
  params <- str_match(file_path, "hierarchical_orders_(.*)_(.*)_(.*)\\.txt")[,2:4]
  
  # Read the file
  data <- read.csv(file_path, sep=",")
  
  # Add parameters and row number to the data
  data %>%
    mutate(
      param1 = as.numeric(params[1]),
      param2 = as.numeric(params[2]),
      param3 = as.numeric(params[3]),
      x = row_number()
    ) %>%
    arrange(ee)
}

# Read all files and combine data
all_data <- map_dfr(file_names, read_file_with_params)

# Create the plot
plot <- ggplot(all_data, aes(x=x, y=ee, color=is_well_nested)) +
  geom_point() +
  facet_wrap(~param1 + param2 + param3, scales = "free") +
  theme_minimal() +
  labs(x = "Index", y = "ee", color = "Is Well Nested") +
  theme(legend.position = "bottom")

# Display the plot
#print(plot)

# Optionally, save the plot
# ggsave("faceted_plot.png", plot, width = 15, height = 10, dpi = 300)

# Create the plot with removed facet labels
plot <- ggplot(all_data, aes(x=x, y=ee, color=is_well_nested)) +
  geom_point() +
  facet_wrap(~param1 + param2 + param3, scales = "free", labeller = function(x) "") +
  theme_minimal() +
  labs(x = "Index", y = "ee", color = "Is Well Nested") +
  theme(legend.position = "bottom",
        strip.background = element_blank(),
        strip.text.x = element_blank())

# Display the plot
#print(plot)

# Optionally, save the plot
# ggsave("faceted_plot.png", plot, width = 15, height = 10, dpi = 300)



library(dplyr)
library(tidyr)
library(ggplot2)
library(purrr)
library(stringr)

# Get all relevant file names
file_names <- list.files(path = "../results", 
                         pattern = "hierarchical_orders_.*\\.txt$", 
                         full.names = TRUE)

# Function to read a file and extract parameters from filename
read_file_with_params <- function(file_path) {
  # Extract parameters from filename
  params <- str_match(file_path, "hierarchical_orders_(.*)_(.*)_(.*)\\.txt")[,2:4]
  
  # Read the file
  data <- read.csv(file_path, sep=",")
  
  # Add parameters and row number to the data
  data %>%
    mutate(
      alpha = round(as.numeric(params[1]),2),
      beta = round(as.numeric(params[2]),2),
      gamma = round(as.numeric(params[3]),2),
      x = row_number()
    ) %>%
    arrange(ee)
}

# Read all files and combine data
all_data <- map_dfr(file_names, read_file_with_params)

all_data = all_data %>% filter((alpha > beta*2) & (beta > gamma*2))

# Create the plot with removed facet labels
plot <- ggplot(all_data, aes(x=x, y=ee, color=is_well_nested, group=paste(alpha, beta, gamma))) +
  geom_line(color="gray") +
  geom_point() +
  theme_minimal() +
  labs(x = "Index", y = "EE", color = "Is Well Nested") +
  theme(legend.position = "bottom",
        strip.background = element_blank(),
        strip.text.x = element_blank())

# Display the plot
plot
ggsave("figures/hierarchical_samples.pdf", plot)


# Optionally, save the plot
# ggsave("faceted_plot.png", plot, width = 15, height = 10, dpi = 300)





