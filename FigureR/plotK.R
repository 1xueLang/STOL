library(ggplot2)
library(ggh4x)
library(showtext)
font_add("Times New Roman", "times.ttf")
showtext_auto()

make.df <- function(value, k, tag) {
    df <- data.frame(
        Value = value,
        Method = rep(c(k), 1, 200),
        Tag = rep(c(tag), 1, 200),
        Epoch = 1:200
    )
    return(df)
}

K1 <- read.csv("../chart/K/run-k-1-tag-Acc.csv")
K2 <- read.csv("../chart/K/run-k-2-tag-Acc.csv")
K3 <- read.csv("../chart/K/run-k-3-tag-Acc.csv")
K4 <- read.csv("../chart/K/run-k-4-tag-Acc.csv")
K5 <- read.csv("../chart/K/run-k-5-tag-Acc.csv")
K6 <- read.csv("../chart/K/run-k-6-tag-Acc.csv")

df <- data.frame()

df <- rbind(df, make.df(K1$Value, "k=1", "Accuracy"))
df <- rbind(df, make.df(K2$Value, "k=2", "Accuracy"))
df <- rbind(df, make.df(K3$Value, "k=3", "Accuracy"))
df <- rbind(df, make.df(K4$Value, "k=4", "Accuracy"))
df <- rbind(df, make.df(K5$Value, "k=5", "Accuracy"))
df <- rbind(df, make.df(K6$Value, "k=6", "Accuracy"))

summary(df)
scaleFUN <- function(x) sprintf("%.0f", x)

ggplot(df, aes(Epoch, Value * 100, color = as.factor(Method))) +
    geom_point() +
    geom_smooth(se = FALSE, span = 0.8) +
    scale_color_manual(values = c("#2b9eb3", "#80ce87", "#44af69", "#d9b6db", "#fcab10", "#f8333c")) +
    scale_x_continuous(limits = c(125, 200), expand = c(0, 0)) +
    scale_y_continuous(limits = c(90, 92), labels = scaleFUN, expand = c(0, 0), breaks = seq(90, 92, 2 / 2)) +
    theme_bw() +
    theme(
        text = element_text(family = 'Times New Roman'),
        # plot.background = element_rect(color = "black", fill = NA, linewidth = 1),
        plot.margin = margin(5, 10, 2, 1),
        # axis.title.y = element_blank(),
        axis.text = element_text(size = 10.5, family = 'Times New Roman'),
        axis.title = element_text(size = 11, family = 'Times New Roman'),
        #
        panel.grid.major = element_line(color = "#adb5bd", linetype = "dotdash"),
        panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank(),
        # panel.border = element_rect(color = "black", linewidth = 1),

        panel.border = element_blank(),
        # axis.line = element_line(colour = "black"),
        #
        strip.background = element_blank(),
        strip.text = element_text(size = 11, family = 'Times New Roman'),
        #
        # legend.background = element_rect(color = "black", size = 0.25),
        legend.background = element_blank(),
        legend.title = element_blank(),
        legend.position = c(0.75, 0.15),
        legend.text = element_text(size = 10, family = 'Times New Roman'),
        legend.direction = "horizontal",
    ) +
    xlab("Epoch") +
    ylab("Accuracy/%") 

ggsave("pdfs/K.pdf", width = 5, height = 3)
