library(dplyr)
library(caret)
library(randomForest)
library(MLmetrics)
library(car)

df <- read.csv('Code/BTL_TKTN/dataset/mlc_churn.csv', stringsAsFactors = FALSE)

df$churn <- ifelse(df$churn == 'yes', 1, 0)
y_target <- as.factor(df$churn) 

cat_cols <- sapply(df, function(x) is.character(x) || is.factor(x))
for (col in names(df)[cat_cols]) {
  df[[col]] <- as.numeric(as.factor(df[[col]])) - 1
}

num_df <- df %>% select(-churn)
corr_matrix <- abs(cor(num_df, use = "complete.obs"))

df <- subset(df, select=-c(total_day_charge, total_eve_charge, total_night_charge, total_intl_charge, voice_mail_plan))

X <- subset(df, select=-c(churn))

k_values <- c(3, 5, 10)
f1_res_crd <- data.frame()

for (k in k_values) {
  set.seed(1234)
  folds <- createMultiFolds(y, k = k, times = 10)
  for (i in seq_along(folds)) {
    train_idx <- folds[[i]]
    X_train <- X[train_idx, ]
    X_test <- X[-train_idx, ]
    y_train <- y[train_idx]
    y_test <- y[-train_idx]
    
    rf <- randomForest(x = X_train, y = y_train, ntree = 100)
    y_pred <- predict(rf, X_test)
    f1 <- F1_Score(y_pred = as.numeric(as.character(y_pred)), 
                   y_true = as.numeric(as.character(y_test)), positive = "1")
    
    f1_res_crd <- rbind(f1_res_crd, data.frame(k = k, f1_score = f1))
  }
}

dir.create("Code/BTL_TKTN/results/crd", recursive = TRUE, showWarnings = FALSE)
write.csv(f1_res_crd, 'Code/BTL_TKTN/results/crd/f1_res_crd.csv', row.names = FALSE)

f1_res_crd$k_factor <- as.factor(f1_res_crd$k)

levene_res <- car::leveneTest(f1_score ~ k_factor, data = f1_res_crd)
p_levene <- levene_res$`Pr(>F)`[1]

model_aov <- aov(f1_score ~ k_factor, data = f1_res_crd)
model_summary <- summary(model_aov)

tukey <- TukeyHSD(model_aov, conf.level = 0.95)

file_conn <- file('Code/BTL_TKTN/results/crd/stat_analysis.txt', open="w")

if (!is.na(p_levene) && p_levene > 0.05) {
  writeLines("p-value > 0.05: Phương sai giữa các nhóm k đồng nhất.\n", file_conn)
} else {
  writeLines("p-value < 0.05: Phương sai giữa các nhóm k khác biệt.\n", file_conn)
}

capture.output(model_summary, file = file_conn)
capture.output(tukey, file = file_conn)
close(file_conn)

png('Code/BTL_TKTN/results/crd/crd_tukey_plot.png', width=800, height=600, res=150)
par(mar=c(5, 4, 4, 2) + 0.1) 

plot(tukey, las=1, col.main="black") 
dev.off()
