library(ggplot2)
library(ggh4x)
library(showtext)
font_add("Times New Roman", "times.ttf")
showtext_auto()

make.df <- function(value, m, tag) {
    df <- data.frame(
        Value = value,
        Method = rep(c(m), 1, 200),
        Tag = rep(c(tag), 1, 200),
        Epoch = 1:200
    )
    return(df)
}

LossS <- read.csv("../chart/run-logs_CIFAR10_msresnet18_6_s-tag-Loss.csv")
AccS <- read.csv("../chart/run-logs_CIFAR10_msresnet18_6_s-tag-acc.csv")
LossT <- read.csv("../chart/run-logs_CIFAR10_msresnet18_6_t-tag-Loss.csv")
AccT <- read.csv("../chart/run-logs_CIFAR10_msresnet18_6_t-tag-acc.csv")
LossO <- read.csv("../chart/run-logs_CIFAR10_msresnet18_6_0-tag-Loss.csv")
AccO <- read.csv("../chart/run-logs_CIFAR10_msresnet18_6_0-tag-acc.csv")
AccF <- read.csv("../chart/run-LOGS_CIFAR-10_ms_resnet18_FullHG-tag-Acc.csv")
LossF <- read.csv("../chart/run-LOGS_CIFAR-10_ms_resnet18_FullHG-tag-Loss.csv")
LossBPTT <- read.csv("../chart/run-CIFAR-10_ms_resnet18_bptt-tag-Loss.csv")
AccBPTT <- read.csv("../chart/run-CIFAR-10_ms_resnet18_bptt-tag-acc.csv")
LossBPTT <- rbind(LossBPTT, tail(LossBPTT, 1) + c(0, 1, 0))
AccBPTT <- rbind(AccBPTT, tail(AccBPTT, 1) + c(0, 1, 0))

df <- data.frame()

df <- rbind(df, make.df(log(LossS$Value), "STOL-S", "Loss"))
df <- rbind(df, make.df(log(LossT$Value), "STOL-T", "Loss"))
df <- rbind(df, make.df(log(LossO$Value), "STOL-O", "Loss"))
df <- rbind(df, make.df(log(LossF$Value), "FHG", "Loss"))
df <- rbind(df, make.df(log(LossBPTT$Value), "BPTT", "Loss"))

df <- rbind(df, make.df(AccS$Value, "STOL-S", "Accuracy"))
df <- rbind(df, make.df(AccT$Value, "STOL-T", "Accuracy"))
df <- rbind(df, make.df(AccO$Value, "STOL-O", "Accuracy"))
df <- rbind(df, make.df(AccF$Value, "FHG", "FHG Accuracy"))
df <- rbind(df, make.df(AccBPTT$Value, "BPTT", "Accuracy"))

summary(df)
scaleFUN <- function(x) sprintf("%.2f", x)

ggplot(df, aes(Epoch, Value, color = Method)) +
    geom_line() +
    scale_color_manual(values = c("#f8333c", "#fcab10", "#44af69", "#987284", "#2b9eb3")) +
    facet_grid(vars(Tag), scales = "free", labeller = as_labeller(c("Loss" = "Log Loss", 'FHG Accuracy' = 'FHG Accuracy', "Accuracy" = "Accuracy"))) +
    facetted_pos_scales(
        y = list(
            Tag == "Loss" ~ scale_y_continuous(limits = c(-3, 0.5), labels = scaleFUN, expand = c(0, 0), breaks = seq(-3, 0.5 - 3.5 / 3, 3.5 / 3)),
            Tag == "Accuracy" ~ scale_y_continuous(limits = c(0.86, 0.95), labels = scaleFUN, expand = c(0, 0), breaks = seq(0.86, 0.95 - 0.09 / 3, 0.09 / 3)),
            Tag == "FHG Accuracy" ~ scale_y_continuous(limits = c(0.3, 0.6), labels = scaleFUN, expand = c(0, 0), breaks = seq(0.3, 0.6 - 0.3 / 3, 0.3 / 3))
        ),
        x = list(
            scale_x_continuous(limits = c(50, 200), expand = c(0, 0))
        )
    ) +
    theme_bw() +
    theme(
        text = element_text(family = 'Times New Roman'),
        plot.background = element_rect(color = "black", fill = NA, linewidth = 1),
        plot.margin = margin(5, 1, 2, 1),
        axis.title.y = element_blank(),
        axis.text = element_text(size = 10.5, family = 'Times New Roman'),
        axis.title = element_text(size = 11, family = 'Times New Roman'),
        #
        panel.grid.major = element_line(color = "#adb5bd", linetype = "dotdash"),
        panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.border = element_rect(color = "black", linewidth = 1),
        #
        strip.background = element_blank(),
        strip.text = element_text(size = 11, family = 'Times New Roman'),
        #
        legend.background = element_rect(color = "black"),
        legend.title = element_blank(),
        legend.position = c(0.85, 0.7),
        legend.text = element_text(size = 10, family = 'Times New Roman'),
    )

ggsave("pdfs/training.pdf", width = 5, height = 4)
