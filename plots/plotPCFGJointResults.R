library(dplyr)
library(ggplot2)

combine_csv_files <- function(directory) {
  files <- list.files(directory, pattern = "pcfg_orders_joint_.*_.*\\.txt", full.names = TRUE)
  
  data_list <- list()
  
  for (file in files) {
    file_name <- basename(file)
    parts <- unlist(strsplit(gsub("pcfg_orders_joint_|\\.txt", "", file_name), "_"))
    seed <- parts[1]
    inverse_temp <- parts[2]
    
    data <- read.csv(file, header = TRUE)
    
    data$Seed <- seed
    data$InverseTemperature <- inverse_temp
    
    data_list[[length(data_list) + 1]] <- data
  }
  
  combined_data <- bind_rows(data_list)
  
  return(combined_data)
}

# Usage
directory <- "../results"
combined_data <- combine_csv_files(directory)

combined_data = combined_data %>% mutate(InverseTemperature = as.numeric(InverseTemperature))

combined_data =  combined_data %>% filter(perm != "general_0")

combined_data$matches_structure = ifelse((combined_data$perm == "(0, 1, 2, 3, 4, 5)"), "local&systematic", ifelse(grepl("general_", combined_data$perm), "unsystematic", "systematic"))


# Assuming combined_data is already defined

seedsByTemperature = unique(combined_data %>% select(Seed, InverseTemperature)) %>% group_by(InverseTemperature) %>% mutate(SeedByTemperature = row_number())
combined_data = merge(combined_data, seedsByTemperature, by=c("InverseTemperature", "Seed"))

# Separate data for "local&systematic"
local_systematic_data <- combined_data %>%
  filter(matches_structure == "local&systematic")

# Data for other levels
other_data <- combined_data %>%
  filter(matches_structure != "local&systematic")


# Define a function to calculate dynamic bar width based on the range of ee values within each facet
calculate_bar_width <- function(data, prop = 0.02) {
  data %>%
    group_by(InverseTemperature, SeedByTemperature) %>%
    summarise(min_ee = min(ee, na.rm = TRUE),
              max_ee = max(ee, na.rm = TRUE)) %>%
    mutate(range_ee = max_ee - min_ee,
           bar_width = range_ee * prop)
}

# Calculate the dynamic bar width for local_systematic_data and other_data
bar_width_data_local <- calculate_bar_width(local_systematic_data)
bar_width_data_other <- calculate_bar_width(other_data)

# Merge the bar width data back to the original datasets
local_systematic_data <- local_systematic_data %>%
  left_join(bar_width_data_local, by = c("InverseTemperature", "SeedByTemperature"))

other_data <- other_data %>%
  left_join(bar_width_data_other, by = c("InverseTemperature", "SeedByTemperature"))

# Plot
plot <- ggplot() +
  geom_bar(data = local_systematic_data %>% 
             group_by(InverseTemperature, SeedByTemperature, ee, matches_structure, bar_width) %>%
             summarise(count = n()), 
           aes(x = ee/log(2), y = count, fill = matches_structure, color = matches_structure, width = bar_width), 
           stat = "identity", position = "identity", alpha = 0.5) +
  geom_density(data = other_data %>%
                 group_by(InverseTemperature, SeedByTemperature) %>%
                 arrange(InverseTemperature, SeedByTemperature, ee) %>%
                 mutate(index = row_number()),
               aes(x = ee/log(2), y = ..scaled.., color = matches_structure, fill = matches_structure, group = paste(SeedByTemperature, matches_structure)),
               alpha = 0.5) +
  facet_grid(SeedByTemperature ~ as.numeric(InverseTemperature), scales = "free_x") +
  labs(x = "Excess Entropy (bits)", y = "Density", color = "Matches Structure", fill = "Matches Structure") +
  theme_minimal() +   theme(
        axis.text.y=element_blank(),
        axis.ticks.x=element_blank())

ggsave("figures/plotPCFGJointResults.pdf", plot, width=10, height=6)

