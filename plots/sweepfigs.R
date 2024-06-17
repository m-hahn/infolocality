rm(list=ls())
setwd("~/projects/infolocality/")
library(tidyverse)
library(latex2exp)
library(ggplot2)
library(plotrix)
library(pracma)
library(ggrepel)

LOG2 = log(2)

### THREE-BIT CODES, INDUCING POSITIVE CORRELATIONS

d = read_csv("output/three_sweep_nonpositional_20240313_with_detection.csv") %>% 
  select(-`...1`) %>%
  mutate(mi=if_else(is.na(mi), LOG2, mi))

codes = read_csv("data/three_codes.csv") %>% select(-`...1`)

weird ="[[0 0 0]\n [1 1 1]\n [1 0 1]\n [0 1 1]\n [0 1 0]\n [1 0 0]\n [1 1 0]\n [0 0 1]]"

h = max(filter(d, mi == 0, t == max(t))$h_t)

d %>%
  filter(mi == 0) %>%
  filter(type == c("id(1) id(2) id(3)") | code == weird) %>%
  mutate(type=factor(type, levels=c("id(1) id(2) id(3)", "holistic"))) %>%
  ggplot(aes(x=t+1, y=h_t/LOG2, color=type)) +
    geom_hline(yintercept=h/LOG2, color="black", linetype="dashed") +
    geom_line(size=1) +
    geom_point(size=3) +
    labs(x=TeX('$n$-gram Order'), y=TeX('$n$-gram Entropy Rate (bits)')) +
    guides(color="none") +
    ylim(1/2,NA) +
    theme_classic()

ggsave("plots/simple_syst_vs_holistic.pdf", width=4.5, height=3.5)



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
    geom_jitter(alpha=1, size=2) +
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

ggsave("plots/perms_by_syst.pdf", width=4.5, height=3.5)


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

ggsave("plots/perms_by_syst_zoom.pdf", width=3, height=2.5)


nonlocal_natural = "[[0 0 0]\n [0 0 1]\n [1 0 1]\n [1 0 0]\n [0 1 0]\n [0 1 1]\n [1 1 1]\n [1 1 0]]"
local_unnatural = "[[0 0 0]\n [0 0 1]\n [1 0 0]\n [1 0 1]\n [0 1 1]\n [0 1 0]\n [1 1 1]\n [1 1 0]]"

dmf = d %>%
  filter(t == max(t)) %>%
  mutate(type=if_else(code == nonlocal_natural, "id(2) id(1) xor(2,3)",
              if_else(code == local_unnatural, "id(2) id(1) xor(1,3)", type))) %>%
  filter(type %in% c(
    "id(1) id(2) id(3)", 
    "id(1) cnot(2, 3)", 
    #"id(1) cnot(3, 2)",
    "cnot(1, 2) id(3)",
    "id(3) id(1) id(2)", # NOTE: id(2) id(1) id(3) is mislabeled in the data
    "id(2) id(1) xor(2,3)",
    "id(2) id(1) xor(1,3)",
    "holistic")) %>%
  mutate(label=case_when(
    type == "id(1) id(2) id(3)" ~ "local systematic, L1 L2 L3",
    type == "id(3) id(1) id(2)" ~ "nonlocal, L3 L1 L2",
    type == "id(1) cnot(2, 3)" ~ "natural, L1 L23",
    type == "cnot(1, 2) id(3)" ~ "unnatural, L12 L3",
    type == "id(2) id(1) xor(2,3)" ~ "nonlocal natural",
    type == "id(2) id(1) xor(1,3)" ~ "local unnatural",
    type == "holistic" & systematic == 0 ~ "holistic",
    T ~ "other"
  )) %>%
  group_by(coupling, mi, label) %>%
    summarize(m=mean(H_M_lower_bound),
              upper=max(H_M_lower_bound),
              lower=min(H_M_lower_bound)) %>%
    ungroup() 

label_position = (filter(dmf, mi/LOG2 > .7, mi/LOG2 < .75) %>% pull(mi))[1]

#bads = c("nonlocal, L3 L1 L2", "other")
bads = c("other", "nonlocal natural", "local unnatural")

dmf %>%
  filter(!(label %in% bads)) %>%
  #mutate(label=factor(label, levels=c("local systematic, L1 L2 L3",
  #                                    "natural, L1 L23",
  #                                    "unnatural, L12 L3",
  #                                    "nonlocal, L3 L1 L2",
  #                                    "holistic"))) %>%
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

ggsave("plots/syst_vs_holistic.pdf", width=4.5, height=3.5)


gg_color_hue <- function(n) {
  hues = seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}

hues = gg_color_hue(4)

label_position = (filter(dmf, mi/LOG2 > .16, mi/LOG2 < .17) %>% pull(mi))[1]
dmf %>%
  filter(!(label %in% bads), label != "holistic") %>%
  filter(mi/LOG2 < .2) %>%
  mutate(label=factor(label, levels=c("local systematic, L1 L2 L3", "natural, L1 L23", "unnatural, L12 L3", "nonlocal, L3 L1 L2"))) %>%
  filter(label == "local systematic, L1 L2 L3" | label == "natural, L1 L23" | label == "unnatural, L12 L3") %>%
  ggplot(aes(x=mi/LOG2, y=m/LOG2, color=label)) +
  geom_line() +
  #geom_label_repel(aes(label=label), 
  #                 data=filter(dmf, mi == label_position, !(label %in% bads), label != "holistic")) +
                   #nudge_x = -0.08,  # Adjust these values to move labels farther from the line
                   #nudge_y = 0.02,
                   #arrow = arrow(length = unit(0.02, "npc")),  # Arrow properties
                   #force = 5) +  # Adjust the force of repulsion if needed
  theme_classic() +
  theme(legend.position=c(.3, .8), legend.title=element_blank()) +ral
  #scale_color_manual(values=hues) +
  #xlim(NA, 0.2) +
  #ylim(NA, 2.1) +
  #guides(color=F) +
  labs(x=TeX('Mutual information $I[M_2 : M_3]$ (bits)'), y='Excess Entropy (bits)') +
  ggtitle(TeX("Languages for $p(M) = p(M_1) \\times p(M_2, M_3)"))

ggsave("plots/syst_vs_holistic4.pdf", width=4.5, height=3.5)



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


### APPROXIMATE-IID CODES

# Calculate EMA function
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
    ggtitle(TeX("E. Languages for approximately iid source"))
    
ggsave("plots/combo_strong_weak.pdf", width=4.5, height=3.5)


### TWO-BIT CODES, FULL SIMPLEX SWEEP

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

safelog = function(x) {
  y = log(x)
  ifelse(is.finite(y), y, 0)
}

entropy = function(p) {
  -sum(p*safelog(p))
}

binary_entropy = function(p) {
  -p*safelog(p) - (1-p)*safelog(1-p)
}
              

binary_pearson = function(p00, p01, p10, p11) {
  a = p10 + p11
  b = p01 + p11
  (p11-a*b)/sqrt(a*(1-a)*b*(1-b))
}

d = read_csv("output/two_sweep_20240319.csv") %>% 
  select(-`...1`) %>%
  mutate(a=p10 + p11,
         b=p01 + p11,
         r=binary_pearson(p00, p01, p10, p11),
         pmi00=log(p00) - log(1-a) - log(1-b),
         pmi10=log(p10) - log(a) - log(1-b),
         pmi01=log(p01) - log(1-a) - log(b),
         pmi11=log(p11) - log(a) - log(b)) %>%
  mutate(H1=binary_entropy(a),
         H2=binary_entropy(b),
         H=H1 + H2 - mi) 

  
d %>%
  filter(t == max(t)) %>%
  #filter(H2 == max(H2)) %>%
  mutate(H1=round(H1, 2),
         H2=round(H2, 2)) %>%
  mutate(xor_entropy=-(p01 + p10)*log(p01 + p10) - (p11 + p00)*log(p00 + p11)) %>%
  select(systematic, p00, p01, p10, p11, H1, H2, a, b, mi, ee, xor_entropy) %>%
  group_by(systematic, p00, p01, p10, p11, H1, H2, a, b, mi, xor_entropy) %>%
    summarize(ee=mean(ee)) %>%
    ungroup() %>%
  spread(systematic, ee) %>%
  mutate(diff=`FALSE` - `TRUE`) %>% # preference for systematic
  filter(a >= .5, b >= .5) %>%
  ggplot(aes(x=H1, y=xor_entropy, color=diff)) +
    geom_point(size=2) +
    scale_color_gradient2() +
    facet_wrap(~H2) +
    theme_classic() +
    labs(x="H[M_1] (nats)", y="H[M_1 xor M_2] (nats)")

middle_a = d %>% pull(a) %>% unique() %>% pluck(56)

# Correlations between M_0 and M_1 that increase the probability of the more common outcome
# are favored by fusion; ones that decrease its probability are not so much.

# Plot probabilities for negative-correlation examples
d %>%
  mutate(xor_entropy=-(p01 + p10)*log(p01 + p10) - (p11 + p00)*log(p00 + p11)) %>%
  select(p00, p01, p10, p11, a, b, mi, r, xor_entropy) %>%
  distinct() %>%
  filter(a < .8, a > .79, b < .8, b > .79) %>%
  mutate(i=1:n()) %>%
  mutate(anticorr=p11/a < b) %>%
  gather(outcome, p, -a, -b, -mi, -i, -anticorr, -r, -xor_entropy) %>%
  mutate(M1=(outcome=="p10" | outcome=="p11"),
         M2=(outcome=="p01" | outcome=="p11")) %>%
  mutate(mi=round(mi, 3), r=round(r, 3)) %>%
  ggplot(aes(x=M1, y=M2, fill=p, label=outcome)) +
    geom_tile() +
    geom_text() +
    facet_wrap(~r) +
    theme_classic() 

d %>%
  mutate(xor_entropy=-(p01 + p10)*log(p01 + p10) - (p11 + p00)*log(p00 + p11)) %>%
  select(p00, p01, p10, p11, a, b, mi, r, xor_entropy) %>%
  distinct() %>%
  filter(a < .8, a > .79, b < .8, b > .79) %>%
  mutate(i=1:n()) %>%
  gather(outcome, p, -a, -b, -mi, -i, -r, -xor_entropy) %>%
  mutate(mi=round(mi, 3), r=round(r, 3)) %>%
  ggplot(aes(x=outcome, y=p, label=outcome)) +
    geom_bar(stat="identity") +
    facet_wrap(~r) +
    theme_classic() 

# Plot pmis

d %>%
  mutate(xor_entropy=-(p01 + p10)*log(p01 + p10) - (p11 + p00)*log(p00 + p11)) %>%
  #mutate(pmi00=p00*pmi00, pmi01=p01*pmi01, pmi10=p10*pmi10, pmi11=p11*pmi11) %>%
  select(pmi00, pmi01, pmi10, pmi11, a, b, mi, r, xor_entropy) %>%
    distinct() %>%
    filter(a < .8, a > .79, b < .8, b > .79) %>%
    mutate(i=1:n()) %>%
    gather(outcome, pmi, -a, -b, -mi, -i, -r, -xor_entropy) %>%
    mutate(M1=(outcome=="pmi10" | outcome=="pmi11"),
           M2=(outcome=="pmi01" | outcome=="pmi11")) %>%
    mutate(mi=round(mi, 3), r=round(r, 3)) %>%
  ggplot(aes(x=M1, y=M2, fill=pmi, label=outcome)) +
    geom_tile() +
    geom_text() +
    scale_fill_gradient2() +
    facet_wrap(~r) +
    theme_classic() 



# in the Weird Case, the HIGHER-PROBABILITY OUTCOME HAS NEGATIVE PMI.
# that is, the HIGHEST PROBABILITY MARGINAL OUTCOMES ARE LESS LIKELY TOGETHER THAN SEPARATE.
# -> Negative correlation when higher-probability outcome = 1.


d %>%
  filter(a < .8, a > .79, b < .8, b > .79) %>% # remove symmetries
  select(p00, p01, p10, p11, pmi00, pmi01, pmi10, pmi11, r, mi) %>%
  distinct() %>%
  mutate(i=1:n()) %>%
  gather(measure, value, -i, -r, -mi) %>%
  mutate(p_or_pmi=if_else(str_detect(measure, "pmi"), "pmi", "p")) %>%
  mutate(outcome=str_replace(measure, "[^\\d]*", "")) %>%
  select(-measure) %>%
  spread(p_or_pmi, value) %>%
  ggplot(aes(x=outcome, y=p, fill=pmi)) +
    geom_bar(stat="identity", color="black") +
    scale_fill_gradient2() +
    facet_wrap(~r) +
    theme_classic()
  
d %>%
  filter(t == max(t)) %>%
  mutate(xor_entropy=-(p01 + p10)*log(p01 + p10) - (p11 + p00)*log(p00 + p11)) %>%
  mutate(a=round(a,2), b=round(b,2), anticorr=p11/a < b) %>% # anticorr means p(M_2 = 1 | M_1 = 1) < p(M_2)
  filter(a != 0, b != 0, 
         a != 1, b != 1, 
         a >= .5, b >= .5, 
         a >= b, 
         a < .7, b < .7) %>%
  select(systematic, mi, xor_entropy, ee, code, a, b, anticorr, r, H1, H2) %>%
  ggplot(aes(x=r, y=ee/LOG2, color=systematic, group=interaction(code, anticorr))) +
    geom_line() +
    facet_grid(a~b, scale="free_y") +
    theme_classic() +
    theme(axis.text.x = element_text(angle = 45, hjust=1),
          legend.position="bottom") +
    labs(x="Pearson's r", y="Excess Entropy (bits)")

ggsave("plots/two_sweep.pdf", width=12, height=13)

d %>%
  filter(t == max(t)) %>%
  mutate(xor_entropy=-(p01 + p10)*log(p01 + p10) - (p11 + p00)*log(p00 + p11)) %>%
  mutate(a=round(a,2), b=round(b,2), anticorr=p11/a < b) %>%
  filter(a != 0, b != 0, a != 1, b != 1, a >= .5, b >= .5, a >= b, a < .7, b < .7) %>%
  select(systematic, mi, xor_entropy, ee, code, a, b, anticorr) %>%
  ggplot(aes(x=mi/LOG2, y=ee/LOG2, color=systematic, group=interaction(code, anticorr), linetype=anticorr)) +
  geom_line() +
  facet_grid(a~b) +
  labs(x=TeX("$I[M_1 : M_2]$ (bits)"), y="Excess Entropy (bits)") +
  theme_classic() +
  theme(axis.text.x = element_text(angle = 45, hjust=1),
        legend.position="bottom")

ggsave("plots/two_sweep_mi.pdf", width=12, height=13)


### STRING PERMUTATION ANALYSIS

perm = read_csv("output/permutations_20240126.csv")

perm %>%
  arrange(ee) %>%
  mutate(i=1:n()) %>%
  mutate(type=if_else(is_contiguous, "concatenative", "nonconcatenative")) %>%
  ggplot(aes(x=i, y=ee/LOG2, color=type)) +
    geom_point() +
    theme_classic() +
    labs(x="Permutation", y="Excess Entropy (bits)", color="") +
    ggtitle("F. String permutations for a systematic language") +
    theme(axis.ticks.x=element_blank(), 
          axis.text.x=element_blank(),
          legend.position=c(.7, .4),
          legend.background = element_rect(fill = "transparent"))

ggsave("plots/string_perms.pdf", width=4.5, height=3.5)


zoom_upto = 1500

perm %>%
  arrange(ee) %>%
  mutate(i=1:n()) %>%
  filter(i < zoom_upto) %>%
  mutate(type=if_else(is_contiguous, "concatenative", "nonconcatenative")) %>%
  ggplot(aes(x=i, y=ee/LOG2, color=type)) +
  geom_point() +
  theme_classic() +
  labs(x="", y="", color="") +
  guides(color="none") +
  theme(axis.ticks.x=element_blank(), 
        axis.text.x=element_blank())

ggsave("plots/string_perms_zoom.pdf", width=3.5, height=2.5)

### HIERARCHICAL SOURCE PERMUTATION ANALYSIS

perm = read_csv("output/hierarchy_20240423.csv")
perm %>%
  arrange(ee) %>%
  mutate(i=1:n()) %>%
  mutate(type=if_else(is_well_nested, "well-nested", "ill-nested")) %>%
  mutate(type=factor(type, levels=c("well-nested", "ill-nested"))) %>%
  ggplot(aes(x=i, y=ee/LOG2, color=type)) +
  geom_point() +
  theme_classic() +
  labs(x="Permutation", y="Excess Entropy (bits)", color="") +
  ggtitle("F. String permutations for a systematic language") +
  theme(axis.ticks.x=element_blank(), 
        axis.text.x=element_blank(),
        legend.position=c(.7, .4),
        legend.background = element_rect(fill = "transparent"))

ggsave("plots/hierarchical_string_perms.pdf", width=4.5, height=3.5)


  