rm(list=ls())
setwd("~/projects/infolocality/")
library(tidyverse)
library(latex2exp)
library(ggplot2)
library(plotrix)
library(pracma)
library(ggrepel)
library(TTR)

LOG2 = log(2)

d = read_csv("output/three_sweep_20231211.csv")
#comb = read_csv("output/combinatoriality_independent_20231212.csv")
comb = read_csv("output/combinatoriality_20231212.csv")
perm = read_csv("output/permutations_20240126.csv")


## Systematicity analysis

dm = d %>%
  filter(t == max(t)) %>%
  arrange(-H_M_lower_bound) %>%
  mutate(rank=1:n()) %>%
  mutate(is_systematic=if_else(systematic == 3, "systematic", "nonsystematic"),
         is_systematic=factor(is_systematic, levels=c("systematic", "nonsystematic"))) 

dm %>%
  ggplot(aes(x=rank, y=H_M_lower_bound/LOG2, color=is_systematic)) +
    geom_point() +
    scale_x_reverse() +
    labs(x="Language", y="Excess Entropy (bits)", color="") +
    theme_classic() +
    theme(axis.ticks.x=element_blank(), 
          axis.text.x=element_blank(),
          legend.position=c(.2, .9))
  
ggsave("plots/three_sweep.pdf", width=4, height=2.5)

n = nrow(dm)

dm %>%
  ggplot(aes(x=-rank, y=H_M_lower_bound/LOG2, color=is_systematic)) +
  geom_point() +
  xlim(NA, -n+500) +
  ylim(2, 2.011) +
  labs(x="Language (Mappings)", y="Excess Entropy (bits)", color="") +
  theme_classic() +
  theme(axis.ticks.x=element_blank(), 
        axis.text.x=element_blank(),
        legend.position=c(.2, .9))

ggsave("plots/three_sweep_zoom.pdf", width=4, height=2)

## Contiguity analysis

perm %>%
  arrange(ee) %>%
  mutate(rank=1:n()) %>%
  mutate(is_contiguous=if_else(is_contiguous, "concatenative", "nonconcatenative")) %>%
  ggplot(aes(x=rank, y=ee/LOG2, color=is_contiguous)) +
  geom_point() +
  labs(x="Language (String Permutations)", y="Excess Entropy (bits)", color="") +
  theme_classic() +
  theme(axis.ticks.x=element_blank(), 
        axis.text.x=element_blank(),
        legend.position=c(.23, .9))

ggsave("plots/perm_sweep.pdf", width=4, height=2)



## Strong-systematicity analysis

combm = comb %>%
  filter(t == max(t)) %>%
  group_by(num_parts, order) %>%
  summarize(m=mean(H_M_lower_bound),
            se=std.error(H_M_lower_bound),
            upper=m+1.96*se,
            lower=m-1.96*se) %>%
  ungroup()

comb %>%
  group_by(order, num_parts, t) %>%
    summarize(H_M_lower_bound=mean(H_M_lower_bound),
              h_t=mean(h_t)) %>%
    ungroup() %>%
  inner_join(combm) %>%
  filter(order == "concatenative") %>%
  mutate(Language=case_when(
    num_parts == 1 ~ str_c("1 morpheme, E = ", format(round(m/LOG2, 2), nsmall=2), " bits"),
    num_parts > 1 ~ str_c(num_parts, " morphemes, E = ", format(round(m/LOG2, 2), nsmall=2), " bits")
    )) %>%
  ggplot(aes(x=t+1, y=h_t/LOG2, color=Language)) +
    geom_line() +
    theme_classic() +
    labs(x=TeX("Markov order $k$"), y=TeX("Markov entropy rate $h_k$"), color="Language") +
    xlim(1, NA) #+ theme(legend.position=c(.3, .3))


# It looks a lot better using MS tradeoff instead of E...
comb %>%
  group_by(order, num_parts, t) %>% # Summarize over random samples
    summarize(H_M_lower_bound=mean(H_M_lower_bound),
              h_t=mean(h_t)) %>%
  ungroup() %>%
  inner_join(combm) %>%
  filter(order == "concatenative") %>%
  ggplot(aes(x=H_M_lower_bound/LOG2, y=h_t/LOG2, color=as.character(num_parts))) +
  geom_line() +
  theme_classic() +
  labs(x=TeX("Memory"), y=TeX("Markov entropy rate $h_k$"), color="Language") 

comb %>%
  group_by(tc, num_parts, t) %>%
    summarize(E=max(H_M_lower_bound)) %>%
    ungroup() %>%
  ggplot(aes(x=tc, y=E, color=as.factor(num_parts))) +
    stat_smooth() +
    theme_classic()


# L(m_1 x m_2) = L(m_1) * L(m_2)

# t test that the languages are different

combstats = comb %>%
  group_by(num_parts, order, sample) %>%
  summarize(tc=unique(tc),
            E=max(H_M_lower_bound),
            ms_auc=trapz(H_M_lower_bound, h_t)) %>%
  ungroup()
  
combstats %>%
  select(-ms_auc) %>%
  spread(num_parts, E) %>%
  mutate(advantage=`2`-`8`) %>%
  ggplot(aes(x=tc, y=advantage)) +
    geom_point() +
    theme_classic() +
    stat_smooth(method='lm')
  
t.test(
  filter(combstats, order == "concatenative", num_parts==4) %>% pull(E),
  filter(combstats, order == "concatenative", num_parts==8) %>% pull(E),
  paired=T
)


t.test(
  filter(combstats, order == "concatenative", num_parts==2) %>% pull(E),
  filter(combstats, order == "concatenative", num_parts==4) %>% pull(E),
  paired=T
)
  
t.test(
  filter(combstats, order == "concatenative", num_parts==1) %>% pull(E),
  filter(combstats, order == "concatenative", num_parts==2) %>% pull(E),
  paired=T
)

d = read_csv("output/strong_systematicity_sweep_20240223.csv") 

d %>%
  group_by(type, sample, tc, coupling) %>%
    summarize(E=max(H_M_lower_bound),
              ms=trapz(H_M_lower_bound, h_t)) %>%
    ungroup() %>%
  ggplot(aes(x=tc, y=E, color=coupling)) +
    stat_smooth(method='lm') +
    geom_point() +
    facet_wrap(~type) +
    theme_classic()

d %>%
  group_by(type, sample, tc, coupling) %>%
  summarize(E=max(H_M_lower_bound),
            ms=trapz(H_M_lower_bound, h_t)) %>%
  ungroup() %>%
  ggplot(aes(x=tc, y=ms, color=coupling)) +
  stat_smooth(method='lm') +
  geom_point() +
  facet_wrap(~type) +
  theme_classic()

d %>%
  group_by(type, sample, tc) %>%
  summarize(E=max(H_M_lower_bound),
            ms=trapz(H_M_lower_bound, h_t - min(h_t))) %>%
  ungroup() %>%
  select(type, sample, tc, ms) %>%
  spread(type, ms) %>%
  ggplot(aes(x=tc/LOG2, y=(weak - strong)/LOG2)) +
    geom_hline(yintercept=0, color="blue") +
    geom_point() +
    theme_classic() +
    labs(y="MS AUC (bits^2)", x="Total Correlation (bits)") +
    ggtitle("Relative advantage of strong systematicity over weak systematicity")

d %>%
  group_by(type, sample, tc, coupling) %>%
  summarize(E=max(H_M_lower_bound),
            ms=trapz(H_M_lower_bound, h_t - min(h_t))) %>%
  ungroup() %>%
  select(-E) %>%
  spread(type, ms) %>%
  mutate(diff_strong=holistic - strong, diff_weak = holistic - weak) %>%
  select(sample, tc, coupling, diff_strong, diff_weak) %>%
  gather(which, diff, -sample, -tc, -coupling) %>%
  ggplot(aes(x=tc/LOG2, y=diff/LOG2, color=which)) +
  geom_hline(yintercept=0, color="blue") +
  geom_point() +
  theme_classic() +
  labs(x="Total Correlation (bits)", y="Delta MS (bits^2)") +
  ggtitle("Relative advantage of systematic code as function of TC")
  

# Structure figure as...
# 1. All holistic
# 2. All strongly-systematic (local vs nonlocal)
# 3. All weakly-systematic (local vs. nonlocal)
# 4. All L1 x L23
# 5. All L12 x L3

d = read_csv("output/three_sweep_nonpositional_20240313_with_detection.csv") %>% 
  select(-`...1`) %>%
  mutate(mi=if_else(is.na(mi), LOG2, mi))

codes = read_csv("data/three_codes.csv") %>% select(-`...1`)

dd = d %>%
  filter(mi == 0) %>%
  select(systematic, ee, mi) %>%
  distinct() %>%
  arrange(ee) %>%
  mutate(rank=1:n()) %>%
  mutate(systematicf=factor(systematic, levels=c("3", "2", "1", "0"))) 

offset = 400

dd %>%
  ggplot(aes(x=rank-offset*systematic, y=ee/LOG2, color=systematicf)) +
    geom_hline(yintercept=2, color="blue") +
    geom_jitter(alpha=1) +
    #facet_wrap(~systexmatic, scale="free_y") +
    ylim(2, NA) +
    labs(x="Language", y="Excess Entropy (bits)", color="Systematicity") +
    theme_classic() +
    guides(color = guide_legend(nrow = 1)) +
    theme(axis.ticks.x=element_blank(), 
          axis.text.x=element_blank(),
          legend.position=c(.7, .2),
          legend.background = element_rect(fill = "transparent")) +
    ggtitle(TeX("Languages for $p(M) = p(M_1) \\times p(M_2) \\times p(M_3)$"))

ggsave("plots/perms_by_syst.pdf", width=6, height=3.5)


local_offset = 50
out_to = 250
dd %>%
  ggplot(aes(x=rank-local_offset*systematic, y=ee/LOG2, color=systematicf)) +
  geom_hline(yintercept=2, color="blue") +
  geom_point(alpha=1, size=3) +
  #facet_wrap(~systematic, scale="free_y") +
  xlim(-3*local_offset, out_to) +
  ylim(2, 2.015) +
  labs(x="", y="", color="") +
  theme_classic() +
  guides(color = "none") +
  theme(axis.ticks.x=element_blank(), 
        axis.text.x=element_blank())

ggsave("plots/perms_by_syst_zoom.pdf", width=4, height=2.5)



dm = d %>%
  filter(t == max(t)) %>%
  group_by(coupling, type, strong, mi, systematic) %>%
    summarize(m=mean(H_M_lower_bound),
              upper=max(H_M_lower_bound),
              lower=min(H_M_lower_bound)) %>%
    ungroup() 
  

dmf = d %>%
  filter(t == max(t)) %>%
  filter(type %in% c(
    "id(1) id(2) id(3)", 
    "id(1) cnot(2, 3)", 
    #"id(1) cnot(3, 2)",
    "cnot(1, 2) id(3)",
    "id(2) id(1) id(3)", 
    "holistic")) %>%
  mutate(label=case_when(
    type == "id(1) id(2) id(3)" ~ "fully systematic, L1 L2 L3",
    type == "id(2) id(1) id(3)" ~ "nonlocal, L2 L1 L3",
    type == "id(1) cnot(2, 3)" ~ "natural, L1 L23",
    type == "cnot(1, 2) id(3)" ~ "unnatural, L12 L3",
    type == "holistic" & systematic == 0 ~ "holistic",
    T ~ "other"
  )) %>%
  group_by(coupling, mi, label) %>%
    summarize(m=mean(H_M_lower_bound),
              upper=max(H_M_lower_bound),
              lower=min(H_M_lower_bound)) %>%
    ungroup() 

label_position = (filter(dmf, mi/LOG2 > .7, mi/LOG2 < .75) %>% pull(mi))[1]

bads = c("nonlocal, L2 L1 L3", "other")

dmf %>%
  filter(!(label %in% bads)) %>%
  ggplot(aes(x=mi/LOG2, y=m/LOG2, color=label)) +
  geom_line() +
  geom_label_repel(aes(label=label), 
                   data=filter(dmf, mi == label_position, !(label %in% bads)),
                   nudge_x = -0.5,  # Adjust these values to move labels farther from the line
                   nudge_y = 0.2,
                   arrow = arrow(length = unit(0.02, "npc")),  # Arrow properties
                   force = 10) +  # Adjust the force of repulsion if needed
  #geom_errorbar(aes(ymin=lower/LOG2, ymax=upper/LOG2)) +
  theme_classic() +
  guides(color=F) +
  labs(x=TeX('Mutual information $I[M_2 : M_3]$ (bits)'), y='Excess Entropy (bits)') +
  ggtitle(TeX("Languages for $p(M) = p(M_1) \\times p(M_2, M_3)"))

ggsave("plots/syst_vs_holistic.pdf", width=6, height=3.5)


dm %>%
  filter(type %in% c(
    "id(1) id(2) id(3)", 
    "id(1) cnot(2, 3)", 
    #"id(1) cnot(3, 2)",
    "cnot(1, 2) id(3)",
    "id(2) id(1) id(3)", 
    "holistic")) %>%
  select(mi, type, m) %>%
  spread(type, m) %>%
  gather(type, m, -mi, -`id(1) id(2) id(3)`) %>%
  mutate(diff=m - `id(1) id(2) id(3)`) %>%
  ggplot(aes(x=mi/LOG2, y=-diff/LOG2, color=type)) +
    geom_hline(yintercept=0) +
    geom_line() +
    #geom_errorbar(aes(ymin=lower/LOG2, ymax=upper/LOG2)) +
    labs(x=TeX('Mutual Information $I[M_2 : M_3]$ (bits)'), y=TeX('\\Delta Excess Entropy (bits)')) +
    theme_classic() +
    theme(legend.position="none")

# Best of the form P(X_1) P(X_2) P(X_3 | X_1, X_2)
d %>% 
  select(-detect23, -detect1, -detect2, -detect3) %>%
  filter(t == max(t)) %>%
  inner_join(codes) %>%
  select(code, mi, H_M_lower_bound, detect1, detect2, detect3, detect12, detect13, detect23) %>%
  filter(detect1, detect23) %>%
  mutate(fully_syst=detect1 & detect2 & detect3) %>%
  distinct() %>%
  ggplot(aes(x=mi/LOG2, y=H_M_lower_bound/LOG2, group=code, color=fully_syst)) +
  geom_line() +
  theme_classic() +
  labs(x=TeX("Mutual Information $I[M_2 : M_3]$ (bits)"),
       y=TeX("Excess Entropy (bits)"))

  
codes %>%
  filter(detect1, detect23, !(detect2 & detect3))




# Add category labels to the lines
# Find the last point of each category for labeling
labels = dm %>%
  group_by(type) %>%
  slice(n()-1) %>% 
  ungroup() 
labels[labels$type == "L123",]$mi = 0

dm %>%
  ggplot(aes(x=mi/LOG2, y=m/LOG2, color=type, label=type)) +
  geom_errorbar(aes(ymin=lower/LOG2, ymax=upper/LOG2)) +
  geom_line() +
  geom_label_repel(
    data = labels,
    aes(label = type),
    size = 5, 
    nudge_y=.02,
    #direction="y",
    segment.color=NA
  ) +
  theme_classic() +
  labs(x=TeX("Mutual Information $I[M_2 : M_3]$ (bits)"), 
       y="Excess Entropy (bits)",
       color="") +
  theme(legend.position="none",
        text = element_text(family = "Arial Unicode MS")) +
  ggtitle(TeX("Codes for $p(M) = p(M_1)p(M_2, M_3)"))

# Calculate EMA function, from ChatGPT
ema <- function(prices, n = 10, smoothing = 2) {
  ema <- numeric(length(prices))
  ema[1] <- prices[1]  # Starting EMA value as the first price
  
  for (i in 2:length(prices)) {
    ema[i] <- (prices[i] * (smoothing / (1 + n))) + ema[i-1] * (1 - (smoothing / (1 + n)))
  }
  
  return(ema)
}


d = read_csv("output/comb_sweep_20240314.csv")

d %>%
  group_by(type, tc, sample) %>%
    summarize(E=max(H_M_lower_bound)) %>%
    ungroup() %>%
  arrange(tc) %>%
  group_by(type) %>%
    mutate(ema=ema(E, n=20, smoothing=2)) %>%
    ungroup() %>%
  ggplot(aes(x=tc, y=ema, color=type)) +
    geom_point() +
    stat_smooth() +
    theme_classic() 

d %>%
  group_by(type, tc, sample) %>%
    summarize(E=max(H_M_lower_bound)) %>%
    ungroup() %>%
  spread(type, E) %>%
  mutate(nonsysl_diff=nonsysl-strong,
         weak_diff=weak-strong) %>%
  select(-nonsysl, -strong, -weak) %>%
  gather(type, E, -tc, -sample) %>%
  group_by(type) %>%
    mutate(ema=ema(E, smoothing=2, n=200)) %>%
    ungroup() %>%
  mutate(type=if_else(type == "nonsysl_diff", "Nonsyst. - Strong syst.",
              if_else(type == "weak_diff", "Weak syst. - Strong syst.", "HONK!"))) %>%
  ggplot(aes(x=tc/LOG2, color=type)) +
    geom_point(alpha=.1, aes(y=E/LOG2)) +
    geom_line(size=1, aes(y=ema/LOG2)) +
    geom_hline(yintercept=0, color="blue") +
    theme_classic() +
    labs(x="KL Divergence from iid (bits)",
         y=TeX("\\Delta Excess Entropy (bits)"),
         color="") +
    theme(legend.position=c(.7, .8)) +
    ggtitle(TeX("Languages for variable-length approx. iid source"))
    
ggsave("plots/combo_strong_weak.pdf", width=6, height=3.5)

d %>%
  filter(tc == 0) %>%
  group_by(type, sample) %>%
    summarize(E=max(H_M_lower_bound)/LOG2) %>%
    ungroup() %>%
  group_by(type) %>%
    summarize(m=mean(E),
              se=std.error(E),
              upper=m+1.96*se,
              lower=m-1.96*se,
              bottom=min(E),
              top=max(E)) %>%
    ungroup()
              

  