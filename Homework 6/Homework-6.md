Homework 6
================
Jingyuan Wu
3/28/2022

Goal: Understand and implement a random forest classifier.

Using the “vowel.train” data, and the “randomForest” function in the R
package “randomForest”. Develop a random forest classifier for the vowel
data by doing the following:

1.  Convert the response variable in the “vowel.train” data frame to a
    factor variable prior to training, so that “randomForest” does
    classification rather than regression.

2.  Review the documentation for the “randomForest” function.

3.  Fit the random forest model to the vowel data using all of the 11
    features using the default values of the tuning parameters.

4.  Use 5-fold CV and tune the model by performing a grid search for the
    following tuning parameters: 1) the number of variables randomly
    sampled as candidates at each split; consider values 3, 4, and 5,
    and 2) the minimum size of terminal nodes; consider a sequence (1,
    5, 10, 20, 40, and 80).

5.  With the tuned model, make predictions using the majority vote
    method, and compute the misclassification rate using the
    ‘vowel.test’ data.

``` r
## load packages
library('randomForest')  ## fit random forest
library('dplyr')    ## data manipulation
library('magrittr') ## for '%<>%' operator
library('gpairs')   ## pairs plot
library('viridis')  ## viridis color palette
library('caret') ## for k-fold
```

## Q1

``` r
## load train data
vowel <- 
  read.csv(url(
    'https://hastie.su.domains/ElemStatLearn/datasets/vowel.train'))[,-1]

## convert the response variable to a factor variable
vowel$y = as.factor(vowel$y)
```

## Q3

``` r
fit <- randomForest(y ~ ., data=vowel)
print(fit)          ## summary of fit object
```

    ## 
    ## Call:
    ##  randomForest(formula = y ~ ., data = vowel) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 3
    ## 
    ##         OOB estimate of  error rate: 2.46%
    ## Confusion matrix:
    ##     1  2  3  4  5  6  7  8  9 10 11 class.error
    ## 1  48  0  0  0  0  0  0  0  0  0  0  0.00000000
    ## 2   0 48  0  0  0  0  0  0  0  0  0  0.00000000
    ## 3   0  0 48  0  0  0  0  0  0  0  0  0.00000000
    ## 4   0  0  0 47  0  1  0  0  0  0  0  0.02083333
    ## 5   0  0  0  0 46  1  0  0  0  0  1  0.04166667
    ## 6   0  0  0  0  0 44  0  0  0  0  4  0.08333333
    ## 7   0  0  0  0  1  0 45  2  0  0  0  0.06250000
    ## 8   0  0  0  0  0  0  0 48  0  0  0  0.00000000
    ## 9   0  0  0  0  0  0  1  0 47  0  0  0.02083333
    ## 10  0  0  0  0  0  0  1  0  0 47  0  0.02083333
    ## 11  0  0  0  0  0  1  0  0  0  0 47  0.02083333

## Q4

``` r
## create folds
set.seed(11)
folds <-createMultiFolds(y=vowel$y,k=5)

mtry_candidates = c(3, 4, 5)

## Grid search to tune the model
set.seed(22)
output=rep(NA, length(mtry_candidates))
for (j in 1:length(mtry_candidates)){
  res = rep(NA, 5)
  for(i in 1:5){  
    fold_test <- vowel[folds[[i]],]
    fold_train <- vowel[-folds[[i]],]
    fold_fit = randomForest(y ~ ., data=fold_train, 
                    mtry=mtry_candidates[j])
    fold_predict <- predict(fold_fit, newdata=fold_test)
    res[i] = as.numeric(confusionMatrix(fold_test$y, fold_predict)$overall['Accuracy'])
  }
  output[j] = mean(res)
}  
#output

CV_accuracy_mtry = matrix(output, 1, length(mtry_candidates))
colnames(CV_accuracy_mtry) = paste0("mtry=", mtry_candidates)
colMeans(CV_accuracy_mtry)
```

    ##    mtry=3    mtry=4    mtry=5 
    ## 0.7395907 0.7338799 0.7277479

``` r
names(colMeans(CV_accuracy_mtry)[colMeans(CV_accuracy_mtry) == max(colMeans(CV_accuracy_mtry))])
```

    ## [1] "mtry=3"

``` r
nodessize_candidates = c(1, 5, 10, 20, 40, 80)

set.seed(1234)
output=rep(NA, length(nodessize_candidates))
for (j in 1:length(nodessize_candidates)){
  res = rep(NA, length(folds))
  for(i in 1:length(folds)){  
    fold_test <- vowel[folds[[i]],]
    fold_train <- vowel[-folds[[i]],]
    fold_fit = randomForest(y ~ ., data=fold_train, 
                    mtry=3, nodessize=nodessize_candidates[j])
    fold_predict <- predict(fold_fit, newdata=fold_test)
    res[i] = as.numeric(confusionMatrix(fold_test$y, fold_predict)$overall['Accuracy'])
  }
  output[j] = mean(res)
}  
#output

CV_accuracy_nodessize = matrix(output, 1, length(nodessize_candidates))
colnames(CV_accuracy_nodessize) = paste0("nodessize=", nodessize_candidates)
colMeans(CV_accuracy_nodessize)
```

    ##  nodessize=1  nodessize=5 nodessize=10 nodessize=20 nodessize=40 nodessize=80 
    ##    0.7291689    0.7288666    0.7264223    0.7285085    0.7287758    0.7273544

``` r
names(colMeans(CV_accuracy_nodessize)[colMeans(CV_accuracy_nodessize) == max(colMeans(CV_accuracy_nodessize))])
```

    ## [1] "nodessize=1"

The best number of variables randomly sampled as candidates at each
split is 3, and the best minimum size of terminal node is 1.

## Q5

``` r
# load test data
vowel_test <- 
  read.csv(url(
    'https://hastie.su.domains/ElemStatLearn/datasets/vowel.test'))[,-1]

set.seed(44)
fit <- randomForest(y ~ ., data = vowel, 
                    mtry = 3, nodesize = 1)

vowel_test$y = as.factor(vowel_test$y)

pred_res <- predict(fit, newdata=vowel_test)
#pred_obs = data.frame(prob=pred_res,obs=vowel_test$y)
## Confusion matrix
#confusionMatrix(vowel_test$y, pred_res)
accuracy = as.numeric(confusionMatrix(vowel_test$y, pred_res)$overall['Accuracy'])
error_rate = 1 - accuracy
error_rate
```

    ## [1] 0.4134199

Misclassification rate is about 41%.
