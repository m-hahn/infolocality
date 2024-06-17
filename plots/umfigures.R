rm(list=ls())
setwd("~/projects/infolocality/")
library(tidyverse)
library(latex2exp)
library(ggplot2)
library(plotrix)
library(pracma)
library(patchwork)

LOG2 = log(2)

d = read_csv("output/um_experiments_20240524.csv")

ee = d %>% 
  group_by(lang, type, sample) %>%
    summarize(E=max(H_M_lower_bound)) %>%
    ungroup() 

ea = d %>%
  group_by(lang, type, sample) %>%
    summarize(ms_auc=trapz(H_M_lower_bound, h_t)) %>%
    ungroup() 

e = inner_join(ee, ea) %>%
  mutate(type=case_when(
    type == "dscramble" ~ "nonconcatenative",
    type == "nonsys" ~ "nonsystematic",
    type == "nonsysl" ~ "nonsystematic (L)",
    type == "real" ~ "real"
  )) %>%
  mutate(type=factor(type, levels=c("real",
                                    "nonsystematic (L)", 
                                    "nonsystematic", 
                                    "nonconcatenative")))

em = e %>%
  group_by(lang, type) %>%
    summarize(m=mean(E),
              se=std.error(E),
              upper=m+1.96*se,
              lower=m-1.96*se) %>%
    ungroup()

fake = filter(e, type != "real")
real = filter(e, type=="real") %>% select(lang, E, ms_auc) %>% rename(real_E=E, real_ms=ms_auc)

fake %>% 
  ggplot(aes(x=E/LOG2, color=type)) + 
    geom_freqpoly(bins=30, alpha=1) + 
    facet_wrap(~lang, scale="free_x") + 
    theme_classic() + 
    #geom_vline(aes(xintercept=m/LOG2, color=type), data=em, linetype="dashed") +
    geom_vline(aes(xintercept=real_E/LOG2), data=real, color="black", size=1) +
    labs(x="Excess Entropy (bits)", y="", color="") +
    scale_color_manual(values=c("darkred", "darkgreen", "blue", "black")) +
    theme(axis.text.y=element_blank(),
          axis.ticks.y=element_blank(),
          legend.position="bottom") 

# Get significance. 

stats = fake %>%
  inner_join(real) %>%
  mutate(real_better=real_E < E) %>%
  group_by(lang, type) %>%
    summarize(p=1-mean(real_better)) %>%
    ungroup() 
  
df = inner_join(fake, stats) %>%
  mutate(p_value=sprintf("%.3f", p)) %>%
  mutate(p_value=str_c(type, ", p", if_else(p_value=="0.000", "<0.001", str_c("=", p_value)))) %>%
  mutate(p_value=factor(p_value, levels=unique(p_value)[order(unique(p_value))]))
    
plots = df %>%
  filter(lang != "Arabic (sound)") %>%
  #mutate(lang=if_else(lang == "Arabic (broken)", "Arabic", lang)) %>%
  split(., .$lang) %>% 
  lapply(function(data) {
  the_lang = unique(data$lang)
  the_ylim = ifelse(the_lang == "Arabic (broken)", 2000, "NA")
  the_plot = ggplot(data, aes(x=E/LOG2, color=p_value)) +
    geom_freqpoly(bins=50, alpha=1) + 
    theme_classic() + 
    geom_vline(aes(xintercept=real_E/LOG2), data=filter(real, lang==the_lang), color="black", size=1) +
    labs(x="Excess Entropy (bits)", y="", color="") +
    scale_color_manual(values=c("darkred", "darkgreen", "blue", "black")) +
    ylim(NA, the_ylim) +
    theme(axis.text.y=element_blank(),
          axis.ticks.y=element_blank(),
          legend.background = element_rect(fill = "transparent"),
          legend.position=c(.7, 1)) +
    ggtitle(ifelse(the_lang == "Arabic (broken)", "Arabic", the_lang))
  
  if (the_lang == "Latin") {
    the_plot = the_plot + xlim(1.45, 3.2)
  }
  the_plot
})
wrap_plots(plots)

ggsave("plots/morpho_experiments.pdf", width=9.5, height=4)

wrap_plots(plots, ncol=1)
ggsave("plots/morpho_experiments_vertical.pdf", width=3.75, height=10)

###### DEPENDENCY PAIRS ######

# Below ordered by number of AN pairs.
# Select 18 that are over 500 and also typologically varied.
GOOD_LANGS = c(
  "German", 
  "Russian", 
  "Czech", 
  "Spanish",
  "Norwegian", 
  #"Polish", 
  #"Romanian", 
  "Turkish",
  "Estonian", 
  "French", 
  #"Belarusian", 
  "English", 
  #"Croatian", 
  "Italian", 
  "Finnish",
  #"Catalan", 
  #"Slovenian", 
  #"Dutch", 
  #"Bulgarian", 
  #"Swedish", 
  #"Ukrainian", 
  "Latvian",
  #"Portuguese", 
  #"Serbian", 
  #"Slovak", 
  "Urdu", # 5000
  "Armenian", 
  #"Danish", 
  #"Lithuanian", 
  #"Irish", 
  "Icelandic", 
  #"Galician",
  "Basque", 
  "Indonesian", 
  "Greek" 
  #"Afrikaans", 
  #"Slavic", 
  #"Maltese",
  #"Naija", 
  #"Welsh", "Sorbian", "Gaelic", "Sami", "Sanskrit", 
  #"Faroese", "Vietnamese" # 500
  #"Kazakh", "Erzya", "Manx", "Zyrian"
)

AN = "output/syntax/letter_form_an_ko_20240228.csv"

d = read_csv(VO) %>%
  select(-`Unnamed: 0`, -`...1`)
  
dt = d %>%
  filter(lang %in% GOOD_LANGS) %>%
  group_by(lang, type, sample) %>%
    summarize(H_M_lower_bound=max(H_M_lower_bound, na.rm=T)) %>%
    ungroup() %>%
  select(H_M_lower_bound, type, lang, sample)

real = dt %>%
  filter(type == "real") %>%
  select(-sample) %>%
  distinct() %>%
  rename(real=H_M_lower_bound) %>%
  select(-type)

dt %>%
  filter(type != "real") %>%
  filter(type != "dscramble") %>%
  inner_join(real) %>%
  ggplot(aes(x=H_M_lower_bound/LOG2, color=type)) +
  geom_freqpoly(bins=25) +
  geom_vline(aes(xintercept=real/LOG2)) +
  facet_wrap(~lang, scale="free") +
  theme_classic() +
  labs(x="Excess Entropy (bits)", y="") +
  scale_color_manual(values=c("blue", "darkgreen")) +
  theme(legend.position="none",
        axis.text.x = element_text(angle = 25, hjust=1, size=7),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank())

ggsave("plots/an_figure.pdf", height=4, width=10.5)
  
  
dt %>%
  filter(type != "real") %>%
  inner_join(real) %>%
  mutate(diff=H_M_lower_bound - real) %>%
  group_by(lang, type) %>%
    summarize(m=mean(H_M_lower_bound - real),
              se=std.error(H_M_lower_bound - real),
              p=mean(H_M_lower_bound < real)) %>%
    ungroup() %>%
  filter(type != "dscramble") %>%
  ggplot(aes(x=type, y=m, color=p, label=lang)) +
    geom_hline(yintercept=0) +
    geom_jitter(size=2, width=.1) +
    theme_classic() +
    geom_text_repel()


d = read_csv("output/syntax/mi_form_an_ko_20240228.csv") %>%
  select(-`Unnamed: 0`, -`...1`)

reals = d %>%
  select(mi, type, lang) %>%
  filter(type=="real") %>%
  select(-type) %>%
  rename(real_mi=mi)

d %>%
  filter(lang %in% GOOD_LANGS) %>%
  filter(type != "real") %>%
  inner_join(reals) %>%
  ggplot(aes(x=mi/LOG2)) +
    geom_freqpoly(bins=25, color="blue") +
    geom_vline(aes(xintercept=real_mi/LOG2)) +
    facet_wrap(~lang, scale="free") +
    theme_classic() +
    labs(x="Mutual Information (bits)", y="") +
    theme(legend.position="none",
        axis.text.x = element_text(angle = 25, hjust=1, size=7),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank())
  


forms = read_csv("output/hun_forms.csv") %>% select(-`...1`)
my_features = c(
  'N;NOM;SG', 
  'N;ACC;SG', 
  'N;DAT;SG', 
  'N;AT+ALL;SG', 
  'N;NOM;PL', 
  'N;ACC;PL', 
  'N;DAT;PL', 
  'N;AT+ALL;PL'
)

forms %>% 
  filter(features %in% my_features) %>%
  mutate(features=factor(features, levels=my_features)) %>%
  arrange(features) %>%
  #mutate(features=str_replace(features, "N;", ""),
  #       features=str_replace(features, "AT+ALL", "ALL")) %>%
  mutate(case=str_replace(features, "N;", ""),
         case=str_replace(case, ";SG", ""),
         case=str_replace(case, ";PL", "")) %>%
  ggplot(aes(x=features, y=count, fill=case)) +
    geom_bar(stat="identity") +
    theme_classic() +
    labs(x="", y="", fill="") 
