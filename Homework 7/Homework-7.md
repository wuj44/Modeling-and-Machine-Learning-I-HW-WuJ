Homework 7
================
Jingyuan Wu
4/12/2022

Goal: Get started using Keras to construct simple neural networks Work
through the “Image Classification” tutorial on the RStudio Keras
website.

-   Use the Keras library to re-implement the simple neural network
    discussed during lecture for the mixture data (see nnet.R). Use a
    single 10-node hidden layer; fully connected.

-   Create a figure to illustrate that the predictions are (or are not)
    similar using the ‘nnet’ function versus the Keras model.

-   (optional extra credit) Convert the neural network described in the
    “Image Classification” tutorial to a network that is similar to one
    of the convolutional networks described during lecture on 4/15
    (i.e., Net-3, Net-4, or Net-5) and also described in the ESL book
    section 11.7. See the !ConvNet tutorial on the RStudio Keras
    website.

``` r
library(keras)
```

``` r
fashion_mnist <- dataset_fashion_mnist()

c(train_images, train_labels) %<-% fashion_mnist$train
c(test_images, test_labels) %<-% fashion_mnist$test
```

``` r
class_names = c('T-shirt/top',
                'Trouser',
                'Pullover',
                'Dress',
                'Coat', 
                'Sandal',
                'Shirt',
                'Sneaker',
                'Bag',
                'Ankle boot')
```

``` r
dim(train_images)
```

    ## [1] 60000    28    28

``` r
dim(train_labels)
```

    ## [1] 60000

``` r
train_labels[1:20]
```

    ##  [1] 9 0 0 3 0 2 7 2 5 5 0 9 5 5 7 9 1 0 6 4

``` r
dim(test_images)
```

    ## [1] 10000    28    28

``` r
dim(test_labels)
```

    ## [1] 10000

``` r
library(tidyr)
library(ggplot2)

image_1 <- as.data.frame(train_images[1, , ])
colnames(image_1) <- seq_len(ncol(image_1))
image_1$y <- seq_len(nrow(image_1))
image_1 <- gather(image_1, "x", "value", -y)
image_1$x <- as.integer(image_1$x)

ggplot(image_1, aes(x = x, y = y, fill = value)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "black", na.value = NA) +
  scale_y_reverse() +
  theme_minimal() +
  theme(panel.grid = element_blank())   +
  theme(aspect.ratio = 1) +
  xlab("") +
  ylab("")
```

![](Homework-7_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

``` r
train_images <- train_images / 255
test_images <- test_images / 255
```

``` r
par(mfcol=c(5,5))
par(mar=c(0, 0, 1.5, 0), xaxs='i', yaxs='i')
for (i in 1:25) { 
  img <- train_images[i, , ]
  img <- t(apply(img, 2, rev)) 
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
        main = paste(class_names[train_labels[i] + 1]))
}
```

![](Homework-7_files/figure-gfm/unnamed-chunk-11-1.png)<!-- -->

``` r
model <- keras_model_sequential()
model %>%
  layer_flatten(input_shape = c(28, 28)) %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')
```

``` r
model %>% compile(
  optimizer = 'adam', 
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
)
```

``` r
model %>% fit(train_images, train_labels, epochs = 5, verbose = 2)
```

``` r
score <- model %>% evaluate(test_images, test_labels, verbose = 0)

cat('Test loss:', score['loss'], "\n")
```

    ## Test loss: 0.3467675

``` r
cat('Test accuracy:', score['accuracy'], "\n")
```

    ## Test accuracy: 0.8766

``` r
predictions <- model %>% predict(test_images)
```

``` r
predictions[1, ]
```

    ##  [1] 8.775721e-07 3.721969e-10 3.627086e-07 1.456523e-08 1.108452e-07
    ##  [6] 5.744073e-03 1.548066e-05 1.413962e-02 2.963595e-06 9.800965e-01

``` r
which.max(predictions[1, ])
```

    ## [1] 10

``` r
class_pred <- model %>% predict(test_images)
class_pred[1:20]
```

    ##  [1] 8.775721e-07 3.324055e-06 2.622249e-07 6.539678e-07 1.602339e-01
    ##  [6] 1.444974e-03 6.653879e-05 1.824512e-05 8.690661e-05 5.866138e-07
    ## [11] 1.751067e-06 2.873181e-06 1.674539e-05 3.358662e-04 5.706855e-04
    ## [16] 1.366777e-05 6.878297e-03 2.591962e-03 2.335458e-05 9.457650e-01

``` r
test_labels[1]
```

    ## [1] 9

``` r
par(mfcol=c(5,5))
par(mar=c(0, 0, 1.5, 0), xaxs='i', yaxs='i')
for (i in 1:25) { 
  img <- test_images[i, , ]
  img <- t(apply(img, 2, rev)) 
  # subtract 1 as labels go from 0 to 9
  predicted_label <- which.max(predictions[i, ]) - 1
  true_label <- test_labels[i]
  if (predicted_label == true_label) {
    color <- '#008800' 
  } else {
    color <- '#bb0000'
  }
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
        main = paste0(class_names[predicted_label + 1], " (",
                      class_names[true_label + 1], ")"),
        col.main = color)
}
```

![](Homework-7_files/figure-gfm/unnamed-chunk-22-1.png)<!-- -->

``` r
# Grab an image from the test dataset
# take care to keep the batch dimension, as this is expected by the model
img <- test_images[1, , , drop = FALSE]
dim(img)
```

    ## [1]  1 28 28

``` r
predictions <- model %>% predict(img)
predictions
```

    ##              [,1]         [,2]         [,3]         [,4]         [,5]
    ## [1,] 8.775704e-07 3.721983e-10 3.627086e-07 1.456523e-08 1.108452e-07
    ##             [,6]         [,7]       [,8]       [,9]     [,10]
    ## [1,] 0.005744071 1.548066e-05 0.01413963 2.9636e-06 0.9800965

``` r
# subtract 1 as labels are 0-based
prediction <- predictions[1, ] - 1
which.max(prediction)
```

    ## [1] 10

``` r
class_pred <- model %>% predict(img)
class_pred
```

    ##              [,1]         [,2]         [,3]         [,4]         [,5]
    ## [1,] 8.775704e-07 3.721983e-10 3.627086e-07 1.456523e-08 1.108452e-07
    ##             [,6]         [,7]       [,8]       [,9]     [,10]
    ## [1,] 0.005744071 1.548066e-05 0.01413963 2.9636e-06 0.9800965

``` r
library('rgl')
library('ElemStatLearn')
library('nnet')
library('dplyr')

## load binary classification example data
data("mixture.example")
dat <- mixture.example
```

``` r
dim(dat$x)
```

    ## [1] 200   2

``` r
## 10 hidden nodes
mod_10 <- keras_model_sequential()
mod_10 %>%
  #layer_flatten(input_shape = c(200, 2)) %>%
  layer_dense(units = 10, activation = 'relu') %>%
  layer_dense(units = 2, activation = 'softmax')


mod_10 %>% compile(
  optimizer = 'adam', 
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
)

mod_10 %>% fit(dat$x, dat$y, epochs = 5, verbose = 2)
```

``` r
plot_mixture_data <- function(dat=mixture.example) {
  ## plot points and bounding box
  #x1r <- range(dat$px1)
  #x2r <- range(dat$px2)
  pts <- plot(dat$x[,1], dat$x[,2],
                type="p", 
                col=ifelse(dat$y, "orange", "blue"))
  #lns <- lines3d(x1r[c(1,2,2,1,1)], x2r[c(1,1,2,2,1)], 1)
  
#  if(showtruth) {
    ## draw Bayes (True) classification boundary
    probm <- matrix(dat$prob, length(dat$px1), length(dat$px2))
    cls <- contourLines(dat$px1, dat$px2, probm, levels=0.5)
    pls <- lapply(cls, function(p) 
      lines(p$x, p$y, col='purple', lwd=3))
    ## plot marginal probability surface and decision plane
    #sfc <- surface3d(dat$px1, dat$px2, dat$prob, alpha=1.0,
      #color="gray", specular="gray")
    #qds <- quads3d(x1r[c(1,2,2,1)], x2r[c(1,1,2,2)], 0.5, alpha=0.4,
      #color="gray", lit=FALSE)
#  }
}



## compute and plot predictions
plot_predictions <- function(fit, dat=mixture.example) {
  
  ## create figure
  plot_mixture_data()

  ## compute predictions from nnet
  preds <- predict(fit, dat$xnew, type="class")
  probs <- predict(fit, dat$xnew, type="raw")[,1]
  probm <- matrix(probs, length(dat$px1), length(dat$px2))
  cls <- contourLines(dat$px1, dat$px2, probm, levels=0.5)

  ## plot classification boundary
  pls <- lapply(cls, function(p) 
    lines(p$x, p$y, col='red', lwd=2))
  
  ## plot probability surface and decision plane
  #sfc <- surface3d(dat$px1, dat$px2, probs, alpha=1.0,
                   #color="gray", specular="gray")
  #qds <- quads3d(x1r[c(1,2,2,1)], x2r[c(1,1,2,2)], 0.5, alpha=0.4,
                 #color="gray", lit=FALSE)
}
#nnet_10probs = 
#nnet(x=dat$x, y=dat$y, size=3, entropy=TRUE, decay=0) %>%
  #plot_nnet_predictions
#mod_10 %>% predict_classes(dat$xnew)
```

``` r
mod_10 %>% plot_predictions()
```

![](Homework-7_files/figure-gfm/unnamed-chunk-31-1.png)<!-- -->

``` r
nnet(x=dat$x, y=dat$y, size=10, entropy=TRUE, decay=0) %>% plot_predictions()
```

![](Homework-7_files/figure-gfm/unnamed-chunk-32-1.png)<!-- -->

    ## # weights:  41
    ## initial  value 154.612638 
    ## iter  10 value 98.633814
    ## iter  20 value 86.206202
    ## iter  30 value 77.871970
    ## iter  40 value 74.223219
    ## iter  50 value 70.043344
    ## iter  60 value 64.806247
    ## iter  70 value 62.635979
    ## iter  80 value 62.259013
    ## iter  90 value 62.084291
    ## iter 100 value 61.943165
    ## final  value 61.943165 
    ## stopped after 100 iterations
