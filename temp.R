library(readr)
library(randomForest)
library(ggplot2)
library(dplyr)
library(caret)

mlc_churn <- read_csv("Code/BTL_TKTN/dataset/mlc_churn.csv")
str(mlc_churn)

mlc_churn <- mlc_churn %>%
  mutate_if(is.character, as.factor)

print(table(mlc_churn$churn))

set.seed(1234)

trainIndex <- createDataPartition(mlc_churn$churn, p=0.7, list=FALSE)
train <- mlc_churn[trainIndex,]
test <- mlc_churn[-trainIndex,]
X_train <- train[,-20]
y_train <- train$churn

control <- rfeControl(functions = rfFuncs, 
                      method = "repeatedcv", 
                      repeats = 5,
                      number = 5)

set.seed(1234)
rfe_results <- rfe(
  x = X_train, 
  y = y_train,                          
  sizes = c(1:19),                                  
  rfeControl = control,
)

rfe_results
plot(rfe_results, type = c("g", "o"))

optimal_features <- predictors(rfe_results)
print(optimal_features)
train_data <- train[, c(optimal_features, y_train)]
print(train_data)

n_min <- sum(train_data$churn == 'yes')
set.seed(1234)
rf <- randomForest(churn~., 
                   data=train,
                   type="classification",
                   confusion=TRUE,
                   importance=TRUE)
rf
summary(rf)

pred <- predict(rf, newdata=test)
cm <- confusionMatrix(data=pred, reference=test$churn, mode="prec_recall", positive="yes")
print(cm)

#f1 lan 1 = 0.7896
