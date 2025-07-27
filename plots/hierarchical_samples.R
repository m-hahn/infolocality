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
plot <- ggplot(all_data, aes(x=x, y=ee/log(2), color=is_well_nested, group=paste(alpha, beta, gamma))) +
  geom_line(color="gray") +
  geom_point() +
  theme_minimal() +
  labs(x = "Index", y = "Excess Entropy (bits)", color = "Is Well Nested") +
  theme(legend.position = "bottom",
        strip.background = element_blank(),
        strip.text.x = element_blank())

# Display the plot
plot
ggsave("figures/hierarchical_samples.pdf", plot, height=5, width=7)



