library(ggplot2)
library(showtext)
font_add("Times New Roman", "times.ttf")
showtext_auto()

BPTT <- c(2431, 3081, 4331, 6747, 12029)
S <- c(2483, 2499, 2515, 2515, 2515)
T <- c(2483, 2499, 2515, 2515, 2515)
O <- c(2287, 2303, 2303, 2303, 2303)
timesteps <- c(2, 4, 8, 16, 32)
Methods <- c("BPTT", "STOL-S", "STOL-T", "STOL-O")
Memorys <- rbind(BPTT, S, T, O)

df <- data.frame()

for (i in 1:4) {
    d <- data.frame(
        Method = rep(Methods[i], 5),
        Memory = as.numeric(Memorys[i, ]) / 10000,
        TimeStep = as.numeric(timesteps)
    )
    df <- rbind(df, d)
}

# print(df)

ggplot(df, aes(factor(TimeStep), Memory, fill = factor(Method), )) +
    geom_bar(stat = "identity", position = position_dodge(0.9), width = 0.8) +
    scale_x_discrete(expand = c(0, 0)) +
    scale_y_continuous(expand = c(0, 0)) +
    coord_flip(ylim = c(0, 1.3)) +
    scale_fill_manual(values = c("#2191a5", "#f8333c", "#fcab10", "#44af69")) +
    theme_bw() +
    theme(
        text = element_text(family = 'Times New Roman'),
        plot.background = element_rect(color = "black", fill = NA, linewidth = 1),
        plot.margin = margin(5, 1, 2, 1),
        #
        axis.text = element_text(size = 10.5, family = 'Times New Roman'),
        axis.title = element_text(size = 11, family = 'Times New Roman'),
        #
        #
        panel.grid.major.y = element_blank(),
        panel.grid.minor = element_blank(),
        #
        legend.background = element_rect(linetype = "solid", colour = "black"),
        legend.title = element_blank(),
        legend.position = c(0.8, 0.2),
        legend.text = element_text(size = 10, family = 'Times New Roman'),
        #
        strip.background = element_blank(),
        strip.placement = "outside",
        strip.text = element_text(size = 11, family = 'Times New Roman'),
    ) +
    xlab("Time Steps") +
    ylab("Memory(10G)") +
    facet_grid(vars(Method))

ggsave("pdfs/memusage.pdf", width = 5, height = 4)
