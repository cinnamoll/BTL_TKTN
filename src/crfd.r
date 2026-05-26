library(dplyr)
library(caret)
library(ranger) 
library(MLmetrics)

df <- read.csv('data/mlc_churn.csv', stringsAsFactors = FALSE)

df$churn <- ifelse(df$churn == 'yes', 1, 0)
y_target <- as.factor(df$churn)

cat_cols <- sapply(df, function(x) is.character(x) || is.factor(x))
for (col in names(df)[cat_cols]) {
  df[[col]] <- as.numeric(as.factor(df[[col]])) - 1
}

num_df <- df %>% select(-churn)
corr_matrix <- abs(cor(num_df, use = "complete.obs"))

to_drop_auto <- c()
high_corr_details <- c()

for (col in colnames(upper)) {
  highly_corr <- rownames(upper)[which(upper[, col] > 0.95)]
  for (row in highly_corr) {
    corr_value <- upper[row, col]
    to_drop_auto <- c(to_drop_auto, col)
    high_corr_details <- c(high_corr_details, 
                           sprintf("tương quan = %.2f với '%s'", col, corr_value, row))
  }
}

to_drop_auto <- unique(to_drop_auto)

cols_to_drop <- c(to_drop_auto, "churn")
X <- df[, !(names(df) %in% cols_to_drop)]

k_values <- c(3, 5, 10)
max_depths <- c(3, 5, 0) 
crfd_results <- data.frame()

dir.create("results/crfd", recursive = TRUE, showWarnings = FALSE)

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
        seed = RANDOM_SEED
      )
      
      y_pred <- predict(rf, data = X_test)$predictions
      f1 <- F1_Score(y_pred = as.numeric(as.character(y_pred)), 
                     y_true = as.numeric(as.character(y_test)), positive = "1")
      
      crfd_results <- rbind(crfd_results, data.frame(
        k = k, 
        max_depth = depth_str, 
        f1_score = f1
      ))
    }
  }
}

write.csv(crfd_results, 'results/crfd/crfd_results.csv', row.names = FALSE)

crfd_results$k_factor <- as.factor(crfd_results$k)
crfd_results$depth_factor <- as.factor(crfd_results$max_depth)

model_crfd <- aov(f1_score ~ k_factor * depth_factor, data = crfd_results)
anova_table <- anova(model_crfd)

file_conn <- file('results/crfd/statistical_analysis.txt', open="w")

capture.output(anova_table, file = file_conn)

capture.output(summary.lm(model_crfd), file = file_conn)
close(file_conn)


png('results/crfd/interaction_plot.png', width=800, height=600, res=150)
interaction.plot(x.factor = crfd_results$k, 
                 trace.factor = crfd_results$max_depth, 
                 response = crfd_results$f1_score, 
                 type = "b", 
                 pch = c(18, 17, 16), 
                 col = c("red", "blue", "green"),
                 xlab = "Số fold (k)", 
                 ylab = "Trung bình F1-Score",
                 trace.label = "max_depth",
                 main = "Đồ thị Tương tác giữa Số fold (k) và Độ sâu cây (max_depth)")
dev.off()

