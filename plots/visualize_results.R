library(data.table)
library(ggplot2)
library(pheatmap)
library(RColorBrewer)
library(xtable)

# Load data
data <- fread("model_modifications.csv")
colnames(data) <- c('Model-Full-Name', 'Accuracy')
data$Model <- data$`Model-Full-Name`
data[Model == "TUnA-no-spectral", Model := "Enc., No Spec. Norm"]
data[Model == "TUnA-unpadded", Model :="No padding"]
data[Model == "TUnA-crossattention", Model :="Enc. Cross"]
data[Model == "TUnA t33", Model :="t33"]
data[Model == "TUnA t36", Model :="t36"]
data[Model == "TUnA t48", Model :="t48"]
data[Model == "D-SCRIPT-ESM-2-encoder-pre-reduction-no-spectral", Model :="Enc., No Spec. Norm"]
data[Model == "D-SCRIPT-ESM-2-encoder-pre-reduction", Model :="Enc. Pre"]
data[Model == "D-SCRIPT-ESM-2 t48", Model :="t48"]
data[Model == "D-SCRIPT-ESM-2-encoder-crossattention", Model :="Enc. Cross"]
data[Model == "D-SCRIPT-ESM-2-encoder-post-reduction", Model :="Enc., Spec. Norm"]
data[Model == "D-SCRIPT-ESM-2 t36", Model :="t36"]
data[Model == "D-SCRIPT-ESM-2 t33", Model :="t33"]
data[Model == "Richoux-ESM-2-encoder-no-spectral", Model :="Enc., No Spec. Norm"]
data[Model == "Richoux-ESM-2-encoder-spectral", Model :="Enc., Spec. Norm"]
data[Model == "Richoux-ESM-2-spectral", Model :="Spec. Norm"]
data[Model == "Richoux-ESM-2 t33", Model :="t33"]
data[Model == "Richoux-ESM-2 t36", Model :="t36"]
data[Model == "Richoux-ESM-2 t48", Model :="t48"]
data[Model == "2d-Crossattention-no-spectral", Model :="Enc., No Spec. Norm"]
data[Model == "2d-Crossattention-encoder-pre-reduction", Model :="Enc. Pre"]
data[Model == "2d-Crossattention t33", Model :="t33"]
data[Model == "2d-Crossattention t36", Model :="t36"]
data[Model == "2d-Crossattention t48", Model :="t48"]
data[Model == "2d-Selfattention-no-spectral", Model :="Enc., No Spec. Norm"]
data[Model == "2d-Selfattention t33", Model :="t33"]
data[Model == "2d-Selfattention t36", Model :="t36"]
data[Model == "2d-Selfattention t48", Model :="t48"]
data[Model == "2d-Selfattention-encoder-pre-reduction", Model := "Enc. Pre"]
data[Model == "2d-baseline t33", Model := "t33"]
data[Model == "2d-baseline t36", Model := "t36"]
data[Model == "2d-baseline t48", Model := "t48"]
data[Model == "RFC-mean t33", Model := "t33"]
data[Model == "RFC-mean t36", Model := "t36"]
data[Model == "RFC-mean t48", Model := "t48"]
data[Model == "RFC-40 t33", Model := "t33"]
data[Model == "RFC-40 t36", Model := "t36"]
data[Model == "RFC-40 t48", Model := "t48"]
data[Model == "RFC-400 t33", Model := "t33"]
data[Model == "RFC-400 t36", Model := "t36"]
data[Model == "RFC-400 t48", Model := "t48"]


data[, `Model Class` := c(
  rep('RFC-40', 3), 
  rep('RFC-400', 3), 
  rep('RFC-mean', 3), 
  rep('2d-baseline', 3), 
  rep('2d-Selfattention', 5),
  rep('2d-Crossattention', 5),
  rep('Richoux-like', 6),
  rep('D-SCRIPT-like', 7),
  rep('TUnA-like', 6)
  )]
data[, `Model Class` := factor(`Model Class`, levels = c(
  'RFC-mean',
  'RFC-40',
  'RFC-400',
  '2d-baseline', 
  '2d-Selfattention',
  '2d-Crossattention',
  'Richoux-like',
  'D-SCRIPT-like',
  'TUnA-like'
  ))]

custom_colors = c(
  "RFC-mean"="#ff0033",
  "RFC-40"="#ff6dbd",
  "RFC-400"="#993366",
  "2d-baseline"="#004949",
  "2d-Selfattention"="#db6d00",
  "2d-Crossattention"="#B2DF8A",
  "Richoux-like"="#FDB462", 
  "D-SCRIPT-like"="#490092",
  "TUnA-like"="#009999"
  )

custom_shapes <- c(
  "RFC-mean"=15,
  "RFC-40"=16,
  "RFC-400"=17,
  "2d-baseline"=18,
  "2d-Selfattention"=7,
  "2d-Crossattention"=8,
  "Richoux-like"=9, 
  "D-SCRIPT-like"=10,
  "TUnA-like"=11
)

data[, Model := factor(Model, 
                       levels = rev(c("t33", "t36", "t48", 
                                  "Enc., Spec. Norm", "Enc., No Spec. Norm",
                                  "Enc. Pre", "Enc. Cross", "Spec. Norm", "No padding")))]
data <- data[order(Model)]
data$Modification <- c(rep("Other", 14), rep("Embedding Size", 41-14))

data <- data[order(Model, Accuracy)]
data[, diffFromLast := Accuracy - shift(Accuracy, n=1, type="lag"), by=Model]
nonJitterPoints <- data[is.na(diffFromLast) | (diffFromLast > 0)]
jitterPoints <- data[which(diffFromLast <= 0)]
nonJitterPoints <- nonJitterPoints[Modification != "Other"]
jitterPoints <- jitterPoints[Modification != "Other"]
nonJitterPoints[Model == "t33", Model := "ESM-2 t33"]
nonJitterPoints[Model == "t36", Model := "ESM-2 t36"]
nonJitterPoints[Model == "t48", Model := "ESM-2 t48"]
jitterPoints[Model == "t33", Model := "ESM-2 t33"]
jitterPoints[Model == "t36", Model := "ESM-2 t36"]
jitterPoints[Model == "t48", Model := "ESM-2 t48"]

nonJitterPoints[, Model := factor(Model, levels = rev(c("ESM-2 t33", "ESM-2 t36", "ESM-2 t48")))]
jitterPoints[, Model := factor(Model, levels = rev(c("ESM-2 t33", "ESM-2 t36", "ESM-2 t48")))]

ggplot() +
  geom_point(size=5, data=nonJitterPoints, aes(x = Accuracy, y = Model, color = `Model Class`, shape = `Model Class`))+
  geom_point(size=5, data=jitterPoints, aes(x = Accuracy, y = Model, color = `Model Class`, shape = `Model Class`), position = position_jitter(height=0.1, width=0, seed=5))+
  scale_shape_manual(values=custom_shapes)+
  xlim(0.5, 0.66)+
  scale_color_manual(values=custom_colors)+
  theme_bw()+
  geom_segment(aes(x = 0.53, xend = 0.53, y = "ESM-2 t33", yend = "ESM-2 t36"), linetype = "longdash", color = "#ff0033")+
  geom_segment(aes(x = 0.53, xend = 0.52, y = "ESM-2 t36", yend = "ESM-2 t48"), linetype = "longdash", color = "#ff0033")+
  geom_segment(aes(x = 0.56, xend = 0.58, y = "ESM-2 t33", yend = "ESM-2 t36"), linetype = "longdash", color = "#ff6dbd")+
  geom_segment(aes(x = 0.58, xend = 0.57, y = "ESM-2 t36", yend = "ESM-2 t48"), linetype = "longdash", color = "#ff6dbd")+
  geom_segment(aes(x = 0.52, xend = 0.52, y = "ESM-2 t33", yend = "ESM-2 t36"), linetype = "longdash", color = "#993366")+
  geom_segment(aes(x = 0.52, xend = 0.52, y = "ESM-2 t36", yend = "ESM-2 t48"), linetype = "longdash", color = "#993366")+
  geom_segment(aes(x = 0.57, xend = 0.53, y = "ESM-2 t33", yend = "ESM-2 t36"), linetype = "longdash", color = "#004949")+
  geom_segment(aes(x = 0.53, xend = 0.52, y = "ESM-2 t36", yend = "ESM-2 t48"), linetype = "longdash", color = "#004949")+
  geom_segment(aes(x = 0.6, xend = 0.57, y = "ESM-2 t33", yend = "ESM-2 t36"), linetype = "longdash", color = "#db6d00")+
  geom_segment(aes(x = 0.57, xend = 0.54, y = "ESM-2 t36", yend = "ESM-2 t48"), linetype = "longdash", color = "#db6d00")+
  geom_segment(aes(x = 0.62, xend = 0.59, y = "ESM-2 t33", yend = "ESM-2 t36"), linetype = "longdash", color = "#B2DF8A")+
  geom_segment(aes(x = 0.59, xend = 0.59, y = "ESM-2 t36", yend = "ESM-2 t48"), linetype = "longdash", color = "#B2DF8A")+
  geom_segment(aes(x = 0.63, xend = 0.64, y = "ESM-2 t33", yend = "ESM-2 t36"), linetype = "longdash", color = "#FDB462")+
  geom_segment(aes(x = 0.64, xend = 0.63, y = "ESM-2 t36", yend = "ESM-2 t48"), linetype = "longdash", color = "#FDB462")+
  geom_segment(aes(x = 0.63, xend = 0.62, y = "ESM-2 t33", yend = "ESM-2 t36"), linetype = "longdash", color = "#490092")+
  geom_segment(aes(x = 0.62, xend = 0.56, y = "ESM-2 t36", yend = "ESM-2 t48"), linetype = "longdash", color = "#490092")+
  geom_segment(aes(x = 0.64, xend = 0.62, y = "ESM-2 t33", yend = "ESM-2 t36"), linetype = "longdash", color = "#009999")+
  geom_segment(aes(x = 0.62, xend = 0.61, y = "ESM-2 t36", yend = "ESM-2 t48"), linetype = "longdash", color = "#009999")+
  guides(color=guide_legend(nrow=2), shape=guide_legend(nrow=2))+
  theme(text = element_text(size=20), 
        axis.title.y = element_blank(),
        legend.position="bottom", 
        legend.direction="vertical",
        legend.margin=margin(),
        legend.justification='left'
        )

ggsave("embedding_sizes.png", width=8, height=4, dpi=300)

encoder_df <- data[`Model Class` %in% c("2d-baseline", "2d-Selfattention", "2d-Crossattention", "D-SCRIPT-like", "Richoux-like", "TUnA-like")]
encoder_df <- encoder_df[!(`Model Class` == "2d-baseline" & Model %in% c("t36", "t48"))]
encoder_df <- encoder_df[!(`Model Class` == "2d-Selfattention" & Model != "t33")]
encoder_df <- encoder_df[!(`Model Class` == "2d-Crossattention" & Model != "t33")]
encoder_df <- encoder_df[!(`Model Class` == "Richoux-like" & !Model %in% c("t36", "Enc., Spec. Norm"))]
encoder_df <- encoder_df[!(`Model Class` == "D-SCRIPT-like" & !Model %in% c("t33", "Enc., Spec. Norm", "Enc. Cross"))]
encoder_df <- encoder_df[!(`Model Class` == "TUnA-like" & !Model %in% c("t33", "Enc. Cross"))]

encoder_df[, Model := ifelse(
  (`Model Class` == "2d-baseline") | 
    (`Model Class` == "Richoux-like" & Model == "t36") | 
    (`Model Class` == "D-SCRIPT-like" & Model == "t33"),
  "Baseline", ifelse(
    (`Model Class` == "2d-Selfattention") | 
      (`Model Class` == "TUnA-like" & Model == "t33") |
      (`Model Class` == "D-SCRIPT-like" & Model == "Enc., Spec. Norm") | 
      (`Model Class` == "Richoux-like" & Model == "Enc., Spec. Norm")
  ,
  "With self-att. Encoder",
  "With cross-att. Encoder")
)]

encoder_df[, Model := factor(Model, levels = rev(c("Baseline", "With self-att. Encoder", "With cross-att. Encoder")))]

ggplot(encoder_df, 
       aes(x = Accuracy, y = Model, color = `Model Class`, shape = `Model Class`))+
  geom_point(size=5)+
  # draw a line from "2d-baseline" to "2d-Selfattention"
  geom_segment(aes(x = 0.57, xend = 0.6, y = "Baseline", yend = "With self-att. Encoder"), linetype = "longdash", color = "#004949")+
  geom_segment(aes(x = 0.6, xend = 0.62, y = "With self-att. Encoder", yend = "With cross-att. Encoder"), linetype = "longdash", color = "#db6d00")+
  geom_segment(aes(x = 0.63, xend = 0.62, y = "Baseline", yend = "With self-att. Encoder"), linetype = "longdash", color = "#490092")+
  geom_segment(aes(x = 0.62, xend = 0.62, y = "With self-att. Encoder", yend = "With cross-att. Encoder"), linetype = "longdash", color = "#490092")+
  geom_segment(aes(x = 0.64, xend = 0.58, y = "Baseline", yend = "With self-att. Encoder"), linetype = "longdash", color = "#FDB462")+
  geom_segment(aes(x = 0.64, xend = 0.66, y = "With self-att. Encoder", yend = "With cross-att. Encoder"), linetype = "longdash", color = "#009999")+
  scale_shape_manual(values=custom_shapes)+
  scale_color_manual(values=custom_colors)+
  xlim(0.5, 0.66)+
  guides(color=guide_legend(nrow=2), shape=guide_legend(nrow=2))+
  theme_bw()+
  theme(text = element_text(size=20), 
        axis.title.y = element_blank(),
        legend.position="bottom", 
        legend.direction="vertical",
        legend.margin=margin(),
        legend.justification='left'
  )

ggsave("add_encoder.png", width=8, height=4, dpi=300)

spec_norm_df <- data[`Model Class` %in% c("TUnA-like", "Richoux-like", "2d-Selfattention", "2d-Crossattention", "D-SCRIPT-like")]
spec_norm_df <- spec_norm_df[!(`Model Class` == "2d-Selfattention" & !Model %in% c("t33", "Enc., No Spec. Norm"))]
spec_norm_df <- spec_norm_df[!(`Model Class` == "2d-Crossattention" & !Model %in% c("t33", "Enc., No Spec. Norm"))]
spec_norm_df <- spec_norm_df[!(`Model Class` == "Richoux-like" & Model %in% c("t33", "t48"))]
spec_norm_df <- spec_norm_df[!(`Model Class` == "D-SCRIPT-like" & !Model %in% c("Enc., Spec. Norm", "Enc., No Spec. Norm"))]
spec_norm_df <- spec_norm_df[!(`Model Class` == "TUnA-like" & !Model %in% c("t33", "Enc., No Spec. Norm"))]

spec_norm_df[`Model Class` == "Richoux-like", `Model Class` := ifelse(
  Model %in% c("t36", "Spec. Norm"), "Richoux-ESM-2", "Richoux-ESM-2-encoder"
)]

spec_norm_df[, Model := ifelse(
  (`Model Class` == "2d-Selfattention" & Model == "t33") | 
    (`Model Class` == "2d-Crossattention" & Model == "t33") | 
    (`Model Class` == "Richoux-ESM-2" & Model == "Spec. Norm") |
    (`Model Class` == "Richoux-ESM-2-encoder" & Model == "Enc., Spec. Norm") |
    (`Model Class` == "D-SCRIPT-like" & Model == "Enc., Spec. Norm") | 
    (`Model Class` == "TUnA-like" & Model == "t33"),
  "With\nSpec. Norm", "Without\nSpec. Norm")]

custom_colors2 = c(
  "RFC-mean"="#ff0033",
  "RFC-40"="#ff6dbd",
  "RFC-400"="#993366",
  "2d-baseline"="#004949",
  "2d-Selfattention"="#db6d00",
  "2d-Crossattention"="#B2DF8A",
  "Richoux-ESM-2"="#FDB462", 
  "Richoux-ESM-2-encoder"="#ffcc00", 
  "D-SCRIPT-like"="#490092",
  "TUnA-like"="#009999"
)

custom_shapes2 <- c(
  "RFC-mean"=15,
  "RFC-40"=16,
  "RFC-400"=17,
  "2d-baseline"=18,
  "2d-Selfattention"=7,
  "2d-Crossattention"=8,
  "Richoux-ESM-2"=9, 
  "Richoux-ESM-2-encoder"=12, 
  "D-SCRIPT-like"=10,
  "TUnA-like"=11
)

spec_norm_df[, Model := factor(Model, levels = c("Without\nSpec. Norm", "With\nSpec. Norm"))]

ggplot(spec_norm_df, 
       aes(x = Accuracy, y = Model, color = `Model Class`, shape = `Model Class`))+
  geom_point(size=5)+
  scale_shape_manual(values=custom_shapes2)+
  scale_color_manual(values=custom_colors2)+
  xlim(0.5, 0.66)+
  guides(color=guide_legend(nrow=2), shape=guide_legend(nrow=2))+
  theme_bw()+
  theme(text = element_text(size=18.5), 
        axis.title.y = element_blank(),
        legend.position="bottom", 
        legend.direction="vertical",
        legend.margin=margin(),
        legend.justification='left'
  )
ggsave("remove_spec_norm.png", width=8, height=3, dpi=300)

encoder_placement <- data[`Model Class` %in% c("2d-Selfattention", "2d-Crossattention", "D-SCRIPT-like")]
encoder_placement <- encoder_placement[!(`Model Class` == "2d-Selfattention" & !Model %in% c("t33", "Enc. Pre"))]
encoder_placement <- encoder_placement[!(`Model Class` == "2d-Crossattention" & !Model %in% c("t33", "Enc. Pre"))]
encoder_placement <- encoder_placement[!(`Model Class` == "D-SCRIPT-like" & !Model %in% c("Enc. Pre", "Enc., Spec. Norm"))]

encoder_placement[, Model := ifelse(
  (`Model Class` == "2d-Selfattention" & Model == "t33") | 
    (`Model Class` == "2d-Crossattention" & Model == "t33") | 
    (`Model Class` == "D-SCRIPT-like" & Model == "Enc., Spec. Norm"),
  "Encoder\npost dim. red.", "Encoder\npre dim. red.")]
encoder_placement[, Model := factor(Model, levels = c("Encoder\npre dim. red.", "Encoder\npost dim. red."))]

ggplot(encoder_placement, 
       aes(x = Accuracy, y = Model, color = `Model Class`, shape = `Model Class`))+
  geom_point(size=5)+
  geom_segment(aes(x = 0.6, xend = 0.56, y = "Encoder\npost dim. red.", yend = "Encoder\npre dim. red."), linetype = "longdash", color = "#db6d00")+
  geom_segment(aes(x = 0.62, xend = 0.58, y = "Encoder\npost dim. red.", yend = "Encoder\npre dim. red."), linetype = "longdash", color = "#B2DF8A")+
  geom_segment(aes(x = 0.62, xend = 0.51, y = "Encoder\npost dim. red.", yend = "Encoder\npre dim. red."), linetype = "longdash", color = "#490092")+
  scale_shape_manual(values=custom_shapes2)+
  scale_color_manual(values=custom_colors2)+
  xlim(0.5, 0.66)+
  guides(color=guide_legend(nrow=1), shape=guide_legend(nrow=1))+
  theme_bw()+
  theme(text = element_text(size=20), 
        axis.title.y = element_blank(),
        legend.position="bottom", 
        legend.direction="vertical",
        legend.margin=margin(),
        legend.justification='left'
  )
ggsave("encoder_placement.png", width=8, height=3, dpi=300)
