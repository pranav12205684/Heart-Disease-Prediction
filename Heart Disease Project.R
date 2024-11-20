library(class)
library(rpart)
library(e1071)
library(caret)
library(caTools)
library(rpart.plot)
library(nnet) 

heart_data <- read.csv("C://Users//Asus//OneDrive//Documents//B Tech//B tech 5 sem//CSE234 Predictive Analytics//heart_dataset.csv")
str(heart_data)
summary(heart_data)

sum(is.na(heart_data))


heart_data$target <- as.factor(heart_data$target)

set.seed(123)
trainIndex <- createDataPartition(heart_data$target, p = 0.8, list = FALSE)
trainData <- heart_data[trainIndex, ]
testData <- heart_data[-trainIndex, ]

# KNN
set.seed(123)
train_control <- trainControl(method = "cv", number = 10)
knn_model <- train(target ~ ., data = trainData, method = "knn",
                   trControl = train_control,
                   tuneGrid = expand.grid(k = c(3, 5, 7, 9, 11)))

knn_model$bestTune
knn_pred <- predict(knn_model, newdata = testData)
cm_knn <- confusionMatrix(knn_pred, testData$target)
acc_knn <- (sum(diag(cm_knn$table)) / sum(cm_knn$table)) * 100
acc_knn
# Accuracy of KNN is 73.52941

# SVM
set.seed(123)
split <- sample.split(heart_data$target, SplitRatio = 0.8)
train_set <- subset(heart_data, split == TRUE)
test_set <- subset(heart_data, split == FALSE)

classifier_svm <- svm(formula = target ~ .,
                      data = train_set,
                      type = 'C-classification',
                      kernel = 'linear')
y_pred_svm <- predict(classifier_svm, newdata = test_set[-14])
cm_svm <- table(test_set$target, y_pred_svm)
acc_svm <- (sum(diag(cm_svm)) / sum(cm_svm)) * 100
acc_svm
# Accuracy of SVM is 86.34146

# Decision Tree
set.seed(678)
sample_indices <- sample(nrow(heart_data), size = 0.8 * nrow(heart_data))
train_data <- heart_data[sample_indices, ]
test_data <- heart_data[-sample_indices, ]
heart_tree <- rpart(target ~ ., data = train_data, method = "class")

plot(heart_tree)
text(heart_tree, use.n = TRUE)
rpart.plot(heart_tree, type = 5, extra = 107)

heart_pred <- predict(heart_tree, test_data, type = "class")
cm_tree <- table(test_data$target, heart_pred)
acc_tree <- sum(diag(cm_tree)) / sum(cm_tree) * 100
acc_tree
# Accuracy of Decision Tree is 85.85366

# ANN
set.seed(456)
heart_data$target <- as.numeric(heart_data$target) - 1
split_ann <- sample.split(heart_data$target, SplitRatio = 0.8)
train_ann <- subset(heart_data, split_ann == TRUE)
test_ann <- subset(heart_data, split_ann == FALSE)

ann_model <- nnet(target ~ ., data = train_ann, size = 5, maxit = 200)

ann_pred_prob <- predict(ann_model, newdata = test_ann, type = "raw")

ann_pred <- ifelse(ann_pred_prob > 0.5, 1, 0)

cm_ann <- table(test_ann$target, ann_pred)
acc_ann <- sum(diag(cm_ann)) / sum(cm_ann) * 100
acc_ann
# Accuracy of ANN is 48.78049

