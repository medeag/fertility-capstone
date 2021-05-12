if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(rpart.plot)) install.packages("rpart.plot", repos = "http://cran.us.r-project.org")

library(caret)
library(data.table)
library(tidyverse)
library(rpart.plot)
options(digits = 5)

#We have used data from https://archive.ics.uci.edu/ml/datasets/Fertility
diagnosis <- read.csv("dataset/fertility_diagnosis.csv", header = TRUE )


dim(diagnosis)
glimpse(diagnosis)

# number of case: normal vs altered
table(diagnosis$Output)


# boxplot for each column
par(mfrow=c(1,5))
for(i in 1:9){
  boxplot(split(diagnosis[,i],diagnosis$Output), main=names(diagnosis)[i])
}
par(mfrow=c(1,1))
hist(diagnosis$Age, main= "Age Frequency" , xlab = "Age")
hist(diagnosis$Alcohol_Consumption, main= "Alcohol Consumption Frequency" , xlab = "Alcohol Consumption")



set.seed(1,sample.kind = "Rounding")
y<-diagnosis$Output
x <- diagnosis[-10]
test_index <- createDataPartition(y, times = 1, p = 0.1, list = FALSE)
test_set_x <- x[test_index, ]
test_set_y <- y[test_index]
train_set_x <- x[-test_index,]
train_set_y <- y[-test_index]

# number of case: normal vs altered in train set
table(train_set_y)

# pca analysis
pca <- prcomp(train_set_x)
summary(pca)
plot(pca, type = "l")




control <- trainControl(method = "cv", number = 10, p = .9)

# LDA 
#set.seed(2007,sample.kind = "Rounding")
train_lda <- train(train_set_x,train_set_y, 
                   method="lda",
                   trControl=control)

# k-Nearest Neighbors
#set.seed(2007,sample.kind = "Rounding")
train_knn <- train(train_set_x,train_set_y,
                   method="knn",  
                   trControl=control)

#Decision Tree 
train_rpart <- train(train_set_x,train_set_y, 
                     method = "rpart",
                     tuneGrid = data.frame(cp = seq(0, 0.05, len = 25)),
                     control = rpart.control(minsplit = 1, minbucket = 1))
rpart.plot(train_rpart$finalModel)

#Random Forest
#set.seed(2007,sample.kind = "Rounding")
train_rf <- train(train_set_x,train_set_y,
                  method="rf", 
                  trControl=control)

varImp(train_rf)


# Evaluate LDA model on test data
predictions <- predict(train_lda, test_set_x)
results <-tibble(method='lda', accuracy=(mean(predictions==test_set_y)))


# Evaluate KNN model on test data
predictions <- predict(train_knn, test_set_x)
results <- results %>% add_row(method = "knn", accuracy=(mean(predictions==test_set_y)))


# Evaluate Decision Tree on Test data
predictions <- predict(train_rf, test_set_x)
results <- results %>% add_row(method = "Decision Tree", accuracy=(mean(predictions==test_set_y)))

# Evaluate Random Forest model on test data
predictions <- predict(train_rf, test_set_x)
results <- results %>% add_row(method = "Random Forest", accuracy=(mean(predictions==test_set_y)))

results

