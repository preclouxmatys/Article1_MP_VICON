# =========================================================
# 07_validity_stats.R
# Stats de validite concomitante MediaPipe vs Vicon :
# QoM (poignet, tete), amplitude (3 plans), angle au coude (3 plans)
# =========================================================

library(readxl)
library(dplyr)
library(tidyr)
library(ggplot2)
library(psych)
library(writexl)
library(scales)

if (!requireNamespace("here", quietly = TRUE)) install.packages("here")
library(here)

FINAL_DIR <- here("data", "final")
FIG_DIR <- here("data", "final", "figures")
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)

# ---------------------------------------------------------
# Fonction generique : r, IC95%, R2, ICC(2,1), RMSE, biais, LoA
# ---------------------------------------------------------
compute_validity <- function(x_mp, x_vicon, label) {
  d <- tibble(MP = x_mp, VICON = x_vicon) %>% drop_na()

  r_test <- cor.test(d$MP, d$VICON, method = "pearson")
  diff <- d$MP - d$VICON
  bias <- mean(diff)
  sd_diff <- sd(diff)

  icc_res <- psych::ICC(d[, c("MP", "VICON")])

  # Test formel de biais proportionnel (Bland-Altman) : regression de la
  # difference (MP - VICON) sur la moyenne des deux systemes. Une pente
  # significativement differente de zero indique un biais proportionnel
  # (l'ecart varie avec la magnitude du mouvement) plutot qu'un biais constant.
  mean_systems <- (d$MP + d$VICON) / 2
  ba_lm <- lm(diff ~ mean_systems)
  ba_slope <- unname(coef(ba_lm)[2])
  ba_slope_p <- summary(ba_lm)$coefficients[2, "Pr(>|t|)"]

  tibble(
    label = label,
    n = nrow(d),
    r = unname(r_test$estimate),
    r_CI_lower = r_test$conf.int[1],
    r_CI_upper = r_test$conf.int[2],
    p_value = r_test$p.value,
    R2 = unname(r_test$estimate)^2,
    ICC_2_1 = icc_res$results[2, "ICC"],
    ICC_lower = icc_res$results[2, "lower bound"],
    ICC_upper = icc_res$results[2, "upper bound"],
    RMSE = sqrt(mean(diff^2)),
    bias_MP_minus_VICON = bias,
    LoA_lower = bias - 1.96 * sd_diff,
    LoA_upper = bias + 1.96 * sd_diff,
    BA_slope = ba_slope,
    BA_slope_p = ba_slope_p
  )
}

scatter_ba_plots <- function(data, mp_col, vicon_col, label, unit_label) {
  d <- data %>%
    select(all_of(mp_col), all_of(vicon_col)) %>%
    rename(MP = all_of(mp_col), VICON = all_of(vicon_col)) %>%
    mutate(mean_systems = (MP + VICON) / 2, diff_MP_minus_VICON = MP - VICON) %>%
    drop_na()

  p_scatter <- ggplot(d, aes(x = VICON, y = MP)) +
    geom_point(alpha = 0.7, size = 2.5) +
    geom_smooth(method = "lm", se = TRUE) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    scale_x_continuous(labels = comma) +
    scale_y_continuous(labels = comma) +
    labs(title = paste(label, ": MediaPipe vs Vicon"), x = paste("Vicon", unit_label), y = paste("MediaPipe", unit_label)) +
    theme_classic(base_size = 14)

  p_ba <- ggplot(d, aes(x = mean_systems, y = diff_MP_minus_VICON)) +
    geom_point(alpha = 0.7, size = 2.5) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    scale_x_continuous(labels = comma) +
    scale_y_continuous(labels = comma) +
    labs(title = paste("Bland-Altman :", label), x = "Moyenne des deux systemes", y = "MediaPipe - Vicon") +
    theme_classic(base_size = 14)

  slug <- gsub("[^a-zA-Z0-9]+", "_", tolower(label))
  ggsave(file.path(FIG_DIR, paste0(slug, "_scatter.png")), p_scatter, width = 7, height = 5, dpi = 300)
  ggsave(file.path(FIG_DIR, paste0(slug, "_bland_altman.png")), p_ba, width = 7, height = 5, dpi = 300)
}

# ---------------------------------------------------------
# 1) QoM : poignet et tete (valeurs normalisees, sans dimension)
# ---------------------------------------------------------
qom <- read_excel(file.path(FINAL_DIR, "comparison_qom_long.xlsx"))

qom_results <- bind_rows(
  compute_validity(qom$QDM_WRIST_MP_norm, qom$QDM_WRIST_VICON_norm, "QoM Poignet"),
  compute_validity(qom$QDM_HEAD_MP_norm, qom$QDM_HEAD_VICON_norm, "QoM Tete")
)
print(qom_results)

scatter_ba_plots(qom, "QDM_WRIST_MP_norm", "QDM_WRIST_VICON_norm", "QoM Poignet", "(ratio norm.)")
scatter_ba_plots(qom, "QDM_HEAD_MP_norm", "QDM_HEAD_VICON_norm", "QoM Tete", "(ratio norm.)")

# ---------------------------------------------------------
# 2) Amplitude poignet (3 plans Vicon)
# ---------------------------------------------------------
amp <- read_excel(file.path(FINAL_DIR, "comparison_amplitude_long.xlsx"))

amp_results <- bind_rows(
  compute_validity(amp$MP_ampWRISTS_pp_mm, amp$VICON_XY_ampWRISTS_pp_mm, "Amplitude XY"),
  compute_validity(amp$MP_ampWRISTS_pp_mm, amp$VICON_XZ_ampWRISTS_pp_mm, "Amplitude XZ"),
  compute_validity(amp$MP_ampWRISTS_pp_mm, amp$VICON_YZ_ampWRISTS_pp_mm, "Amplitude YZ")
)
print(amp_results)

scatter_ba_plots(amp, "MP_ampWRISTS_pp_mm", "VICON_XY_ampWRISTS_pp_mm", "Amplitude XY", "(mm)")
scatter_ba_plots(amp, "MP_ampWRISTS_pp_mm", "VICON_XZ_ampWRISTS_pp_mm", "Amplitude XZ", "(mm)")
scatter_ba_plots(amp, "MP_ampWRISTS_pp_mm", "VICON_YZ_ampWRISTS_pp_mm", "Amplitude YZ", "(mm)")

# ---------------------------------------------------------
# 3) Angle au coude : resume des correlations/RMSE deja calculees par ligne
# ---------------------------------------------------------
angle <- read_excel(file.path(FINAL_DIR, "comparison_elbow_angle_long.xlsx"))

angle_summary <- angle %>%
  filter(is.na(error) | error == "") %>%
  group_by(plane) %>%
  summarise(
    n = n(),
    mean_r_VI2_bilateral = mean(c(corr_MP_vs_VI2_R, corr_MP_vs_VI2_L), na.rm = TRUE),
    sd_r_VI2_bilateral = sd(c(corr_MP_vs_VI2_R, corr_MP_vs_VI2_L), na.rm = TRUE),
    mean_RMSE_VI2_bilateral = mean(c(rmse_MP_vs_VI2_R, rmse_MP_vs_VI2_L), na.rm = TRUE),
    sd_RMSE_VI2_bilateral = sd(c(rmse_MP_vs_VI2_R, rmse_MP_vs_VI2_L), na.rm = TRUE),
    mean_r_VI3_bilateral = mean(c(corr_MP_vs_VI3_R, corr_MP_vs_VI3_L), na.rm = TRUE),
    sd_r_VI3_bilateral = sd(c(corr_MP_vs_VI3_R, corr_MP_vs_VI3_L), na.rm = TRUE),
    mean_RMSE_VI3_bilateral = mean(c(rmse_MP_vs_VI3_R, rmse_MP_vs_VI3_L), na.rm = TRUE),
    sd_RMSE_VI3_bilateral = sd(c(rmse_MP_vs_VI3_R, rmse_MP_vs_VI3_L), na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(mean_RMSE_VI2_bilateral)

print(angle_summary)

p_angle_r <- angle %>%
  select(plane, corr_MP_vs_VI2_R, corr_MP_vs_VI2_L) %>%
  pivot_longer(-plane, names_to = "side", values_to = "r") %>%
  ggplot(aes(x = plane, y = r)) +
  geom_boxplot(outlier.shape = NA) +
  geom_jitter(width = 0.15, alpha = 0.4) +
  labs(title = "Angle au coude : correlation MediaPipe vs Vicon (2D)", x = "Plan de projection Vicon", y = "r (par participant)") +
  theme_classic()
ggsave(file.path(FIG_DIR, "elbow_angle_correlations.png"), p_angle_r, width = 8, height = 5, dpi = 300)

# ---------------------------------------------------------
# 4) Export
# ---------------------------------------------------------
write_xlsx(
  list(qom_results = qom_results, amplitude_results = amp_results, angle_summary = angle_summary),
  file.path(FINAL_DIR, "validity_results_all.xlsx")
)
cat("Sauvegarde :", file.path(FINAL_DIR, "validity_results_all.xlsx"), "\n")
