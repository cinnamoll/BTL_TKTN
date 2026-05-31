library(dplyr)
library(caret)
library(randomForest)
library(MLmetrics)
library(corrplot)

df <- read.csv("Code/BTL_TKTN/dataset/mlc_churn.csv", stringsAsFactors = FALSE)

str(df)

df$churn <- ifelse(df$churn == 'yes', 1, 0)
y_target <- as.factor(df$churn) # Dùng cho bài toán phân loại

cat_cols <- sapply(df, function(x) is.character(x) || is.factor(x))
for (col in names(df)[cat_cols]) {
  df[[col]] <- as.numeric(as.factor(df[[col]])) - 1
}

numeric_vars <- sapply(df, is.numeric)
cor_matrix <- cor(df[, numeric_vars])
cor_matrix
corrplot(cor_matrix, method = "number", addCoef.col = "black", bg="gray", type = "lower", number.digits=2, number.cex=0.5, diag=FALSE)

print(sum(mlc_churn$voice_mail_plan=="no" & mlc_churn$number_vmail_messages>0))

#drop moi cot charges do gia tien duoc sinh tu so phut goi
df <- subset(df, select=-c(total_day_charge, total_eve_charge, total_night_charge, total_intl_charge, voice_mail_plan))

X <- subset(df, select=-c(churn))
y <- y_target


set.seed(1234)
train_idx <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_idx, ]
X_test <- X[-train_idx, ]
y_train <- y[train_idx]
y_test <- y[-train_idx]

set.seed(1234)
rf <- randomForest(x = X_train, y = y_train)

y_pred <- predict(rf, X_test)
f1Score <- F1_Score(y_pred = as.numeric(as.character(y_pred)), 
                 y_true = as.numeric(as.character(y_test)), positive = "1")
print(f1Score)

dir.create("BTL_TKTN/results/baseline", recursive = TRUE, showWarnings = FALSE)

df_baseline <- data.frame(model = 'Baseline Random Forest (M)', f1_score = f1Score)
write.csv(df_baseline, 'BTL_TKTN/results/baseline/baseline_results.csv', row.names = FALSE)

importances <- importance(rf)
df_importance <- data.frame(
  feature = rownames(importances),
  importance = as.numeric(importances[, 1])
)
df_importance <- df_importance[order(-df_importance$importance), ]
write.csv(df_importance, 'BTL_TKTN/results/baseline/feature_importances.csv', row.names = FALSE)


file_conn <- file("BTL_TKTN/results/baseline/dropped.txt", open="w")
lines <- c("total_day_charge", "total_eve_charge", "total_night_charge", "total_intl_charge", "voice_mail_plan")
writeLines(lines, file_conn)
close(file_conn)
