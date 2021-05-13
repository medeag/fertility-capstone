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
#We added header to it and uploaded to the github, from where we are using it.
url_csv <- 'https://raw.githubusercontent.com/medeag/fertility-capstone/main/dataset/fertility_diagnosis.csv'
diagnosis <-read.csv(url_csv, header = TRUE)
#diagnosis <- read.csv("dataset/fertility_diagnosis.csv", header = TRUE )


dim(diagnosis)
glimpse(diagnosis)

# number of case: normal vs altered
table(diagnosis$Output)


# boxplot for each column
par(mfrow = c(3, 3))
for(i in 1:9){
  boxplot(split(diagnosis[,i],diagnosis$Output), main=names(diagnosis)[i])
}

par(mfrow = c(1, 2))
diagnosis_o <- diagnosis %>% filter(Output=='O')
hist(diagnosis_o$Age, main = "Age Frequency (Altered)", xlab = "Age")
diagnosis_n <- diagnosis %>% filter(Output=='N')
hist(diagnosis_n$Age, main = "Age Frequency (Normal)", xlab = "Age")

# Please take in consideration that although analysis was performed on "R version 4.0.2 (2020-06-22)", 
# we have not used sample.kind="Rounding" on the seed,
# since we find it redundant to be compatible with 3.5
set.seed(197379245)
y<-diagnosis$Output
x <- diagnosis[-10]

#split data into train and test sets, with 70% and 30% respectively
test_index <- createDataPartition(y, times = 1, p = 0.3, list = FALSE)
test_set_x <- x[test_index, ]
test_set_y <- y[test_index]
train_set_x <- x[-test_index,]
train_set_y <- y[-test_index]

# number of case: normal vs altered in train set
table(train_set_y)

# pca analysis
pca <- prcomp(train_set_x)
summary(pca)
par(mfrow = c(1, 1))
plot(pca, type = "l")


control <- trainControl(method = "cv", number = 5)

# LDA 
train_lda <- train(train_set_x,train_set_y, 
                   method="lda",
                   trControl=control)

# k-Nearest Neighbors
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
train_rf <- train(train_set_x,train_set_y,
                  method="rf", 
                  trControl=control)

varImp(train_rf)


# Evaluate LDA model on test data
predictions <- predict(train_lda, test_set_x)
results <-tibble(method='LDA', accuracy=(mean(predictions==test_set_y)))


# Evaluate KNN model on test data
predictions <- predict(train_knn, test_set_x)
results <- results %>% add_row(method = "K-NN", accuracy=(mean(predictions==test_set_y)))


# Evaluate Decision Tree model on test data
predictions <- predict(train_rf, test_set_x)
results <- results %>% add_row(method = "Decision Tree", accuracy=(mean(predictions==test_set_y)))

# Evaluate Random Forest model on test data
predictions <- predict(train_rf, test_set_x)
results <- results %>% add_row(method = "Random Forest", accuracy=(mean(predictions==test_set_y)))

#print results
results

#print max result method(s), in theory several method could give the same result (not in our case)
results[which(results$accuracy == max(results$accuracy)),]$method



