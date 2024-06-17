rm(list=ls())
setwd("~/projects/infolocality/")
library(tidyverse)
library(latex2exp)
library(ggplot2)
library(stringr)
library(ggpubr)
library(ggrepel)
library(broom)
library(plotrix)

LOG2 = log(2)

typology = read_csv("output/merged_np_typology.csv") %>%
  select(type, group, af_sum, num_genera_sum)

d = read_csv("output/np_order_de_d1.csv") %>%  
  select(-`...1`) 

if("t" %in% names(d)) {
  d = d %>%
    filter(t == max(t)) %>%
    inner_join(typology)
}
d = d %>%
    group_by(group, af_sum, num_genera_sum) %>%
      summarize(H_M_lower_bound=mean(H_M_lower_bound)) %>%
      ungroup() %>%
    mutate(group_name=str_replace(group, 
      "frozenset\\(\\{\\'(.*)\\', \\'(.*)\\'\\}\\)", "\\1/\\2"))

correlation = with(d, cor.test(log(num_genera_sum), H_M_lower_bound, method="spearman")) %>%
  tidy() %>%
  pull(estimate)

d %>%
  ggplot(aes(x=num_genera_sum, y=H_M_lower_bound/LOG2, label=group_name)) +
    stat_smooth(method='lm') +
    geom_point() +
    geom_text_repel() +
    theme_classic() +
    stat_cor(label.x=1.3, label.y=3.3) +
    scale_x_log10(
      breaks = c(1, 10, 100),
      labels = c(1, 10, 100)
    ) +
    annotation_logticks() +
    labs(x="Frequency across Languages", y="Excess Entropy (bits)") 

ggsave("plots/np_order_corr_de_d1.pdf", width=3.5, height=2.5)

d %>%
  ggplot(aes(x=H_M_lower_bound/LOG2, y=num_genera_sum, label=group_name)) +
  stat_smooth(method='lm') +
  geom_point() +
  geom_text_repel() +
  theme_classic() +
  stat_cor(label.x=3.15, label.y=1.8) +
  scale_y_log10(
    breaks = c(1, 10, 100),
    labels = c(1, 10, 100)
  ) +
  annotation_logticks() +
  labs(y="Frequency across Language Genera", x="Excess Entropy (bits)") +
  ggtitle("Noun Phrase Orders by Excess Entropy")

ggsave("plots/np_order_corr_de1_t.pdf", width=4.5, height=3.5)




