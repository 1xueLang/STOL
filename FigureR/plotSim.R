library(ggplot2)
library(showtext)

font_add("Times New Roman", "times.ttf")
showtext_auto()

df <- read.csv("./csvs/grad_sim.csv", stringsAsFactors = FALSE)

ggplot(df, aes(as.numeric(Layer), as.numeric(Mean1), fill = factor(Method))) +
    geom_ribbon(aes(ymin = Mean1 - Sd1, ymax = as.numeric(Mean1) + as.numeric(Sd1))) +
    geom_line(aes(y = Mean1, color = factor(Method))) +
    geom_point(aes(y = Mean1, color = factor(Method))) +
    facet_grid(cols = vars(Epoch)) +
    scale_fill_manual(values = c("#f8333c22", "#fcab1022", "#44af6922")) +
    scale_color_manual(values = c("#f8333c", "#fcab10", "#44af69")) +
    facet_grid(cols = vars(Epoch), labeller = labeller(Epoch = label_both)) +
    theme_bw() +
    theme(
        plot.margin = margin(1, 1, 1, 0),
        axis.text = element_text(size = 13, family = "Times New Roman"),
        axis.title = element_text(size = 14, family = "Times New Roman"),
        legend.background = element_blank(),
        # legend.background = element_rect(color = "black"),
        legend.title = element_blank(),
        legend.position = c(0.11, 0.15),
        legend.text = element_text(size = 10, family = 'Times New Roman'),
        legend.direction = "horizontal",
        #
        panel.grid.major = element_line(color = "#adb5bd", linetype = "dotdash"),
        panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank(),
        #
        strip.background = element_blank(),
        strip.placement = "outside",
        strip.text = element_text(size = 13, family = "Times New Roman"),
    ) +
    xlab(element_blank()) +
    ylab("Cosine Distance")

ggsave("pdfs/cs-sim.pdf", width = 12, height = 2.5)

ggplot(df, aes(as.numeric(Layer), as.numeric(Mean2), fill = factor(Method))) +
    geom_ribbon(aes(ymin = Mean2 - Sd2, ymax = as.numeric(Mean2) + as.numeric(Sd2))) +
    geom_line(aes(y = Mean2, color = factor(Method))) +
    geom_point(aes(y = Mean2, color = factor(Method))) +
    facet_grid(cols = vars(Epoch), labeller = labeller(Epoch = label_both)) +
    # scale_fill_manual(values = c("#1697a644", "#0e606b44", "#ffc24b44")) +
    # scale_color_manual(values = c("#1697a6", "#0e606b", "#ffc24b")) +
    scale_fill_manual(values = c("#f8333c22", "#fcab1022", "#44af6922")) +
    scale_color_manual(values = c("#f8333c", "#fcab10", "#44af69")) +
    theme_bw() +
    theme(
        plot.margin = margin(1, 1, 1, 0),
        axis.text = element_text(size = 13, family = "Times New Roman"),
        axis.title = element_text(size = 14, family = "Times New Roman"),
        legend.background = element_blank(),
        # legend.background = element_rect(color = "black"),
        legend.title = element_blank(),
        legend.position = c(0.11, 0.85),
        legend.text = element_text(size = 10, family = 'Times New Roman'),
        legend.direction = "horizontal",
        # legend.text = element_text(size = 11),
        #
        panel.grid.major = element_line(color = "#adb5bd", linetype = "dotdash"),
        panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank(),
        #
        strip.background = element_blank(),
        strip.placement = "outside",
        strip.text = element_text(size = 13, family = "Times New Roman"),
    ) +
    xlab(element_blank()) +
    ylab("Euclidean Distance")

ggsave("pdfs/ed-sim.pdf", width = 12, height = 2.5)
