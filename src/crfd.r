library(dplyr)
library(caret)
library(ranger) 
library(MLmetrics)

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
max_depths <- c(3, 5, 0) 
crfd_results <- data.frame()

dir.create("Code/BTL_TKTN/results/crfd", recursive = TRUE, showWarnings = FALSE)

for (k in k_values) {
  set.seed(1234)
  folds <- createMultiFolds(y, k = k, times = 10)
  
  for (depth in max_depths) {
    depth_str <- ifelse(depth == 0, "None", as.character(depth))
    
    for (i in seq_along(folds)) {
      train_idx <- folds[[i]]
      X_train <- X[train_idx, ]
      X_test <- X[-train_idx, ]
      y_train <- y[train_idx]
      y_test <- y[-train_idx]
      
      rf <- ranger(
        x = X_train, 
        y = y_train, 
        max.depth = depth, 
        seed = 1234
      )
      
      y_pred <- predict(rf, data = X_test)$predictions
      f1 <- F1_Score(y_pred = as.numeric(as.character(y_pred)), 
                     y_true = as.numeric(as.character(y_test)), positive = "1")
      
      if (is.na(f1)) {
        f1 <- 0
      }

      crfd_results <- rbind(crfd_results, data.frame(
        k = k, 
        max_depth = depth_str, 
        f1_score = f1
      ))
    }
  }
}

write.csv(crfd_results, 'Code/BTL_TKTN/results/crfd/crfd_results.csv', row.names = FALSE)

crfd_results$k_factor <- as.factor(crfd_results$k)
crfd_results$max_depth <- as.factor(crfd_results$max_depth)

model_crfd <- aov(f1_score ~ k_factor * max_depth, data = crfd_results)
anova_table <- anova(model_crfd)

file_conn <- file('Code/BTL_TKTN/results/crfd/statistical_analysis.txt', open="w")

capture.output(anova_table, file = file_conn)

capture.output(summary.lm(model_crfd), file = file_conn, append=TRUE)
close(file_conn)


png('Code/BTL_TKTN/results/crfd/interaction_plot.png', width=800, height=600, res=150)
par(mar=c(5, 4, 4, 2) + 0.1) 

interaction.plot(
  x.factor = crfd_results$k_factor,
  trace.factor = crfd_results$max_depth, 
  response = crfd_results$f1_score, 
  type = "b", 
  pch = c(18, 17, 16), 
  col = c("red", "blue", "green"),
  xlab = "Số fold (k)", 
  ylab = "Trung bình F1-Score",
  trace.label = "max_depth",
  ylim = c(0, 1), 
  main = "Tương tác giữa Số fold (k) và Độ sâu cây"
)
dev.off()

