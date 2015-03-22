---
title: "Practical Machine Learning Project"
author: "Alljoy_j"
date: "Sunday, March 22, 2015"
output: html_document
---
##Synopsis
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively.  One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, the goal is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

The goal of the project is to predict the manner in which they did the exercise -- either correctly, or which of the 5 incorrect ways was used.

##Data Processing
###Loading and preprocessing the data

The training data for this project are available here: 
  https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here: 
  https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment. 

The data is loaded into the working directory using the below code.  Note that several libraries must be installed prior to running this document, so a function is used to install packages if they haven't already been installed (I didn't write it, it came from stack overflow). Also, the seed set must be used to obtain the same results.  When the document is loaded, the number of observations and the number of variables will display:


```r
# setwd("C:/Data/Coursera/8 PracticalMachineLearning/Week3/")

 ##  Package to check for required package, install it if missing, and load it    
  packageload<-function(x){
    x<-as.character(match.call()[[2]])
    if (!require(x,character.only=TRUE)){
      install.packages(pkgs=x,repos="http://cran.r-project.org")
      require(x,character.only=TRUE)
    }
  }
  
packageload(AppliedPredictiveModeling)
packageload(caret)
packageload(data.table)
packageload(data.table)
packageload(plyr)
packageload(gmb)
```

```
## Warning in install.packages :
##   package 'gmb' is not available (for R version 3.1.0)
```

```r
packageload(randomForest)
packageload(ggplot2)

set.seed(3433) #this number was used in one of our assignments, it worked, didn't change it
trnUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
#reads the dataset from the website and finds the number of rows & columns
trn_ds <- read.csv(url(trnUrl), na.strings=c("NA","#DIV/0!",""))
# count of observations
print (NROW(trn_ds))
```

```
## [1] 19622
```

```r
# count of variables
print (NCOL(trn_ds))
```

```
## [1] 160
```

```r
ds_names<- names(trn_ds)
```


After the data is read, an initial view of the data showed several records with NA's in the variable.  When these were examined, they were found to be aggregate columns.  To prevent issues with collinarity, These will not be used in the analysis as the underlying data they are based off of will be included instead.

Only those fields of interest are subseted into a data frame (named ds_names2), and that frame is divided into two sets (First_trn_ds and Second_trn_ds).  When completed, the number of observations and number of variables for the first data set (the 60% data set) and the number of observations and number of variables for the second data set (the 40% data set) display, using the below code:


```r
#removes aggregated columns
#removes aggregated columns
ds_names2<-names(which(colSums(is.na(trn_ds))==0)) #doesn't account for blanks, only NA's
trn_ds2<-subset(trn_ds, select = ds_names2)
setnames(trn_ds2,ds_names2)
#removes near zero covariets
near_zero <- nearZeroVar(trn_ds2, saveMetrics = TRUE)
trn_ds2<-trn_ds2[,!near_zero$nzv]
ds_names2<-names(trn_ds2)
rem_colct<-NCOL(trn_ds2)  #remaining columns
#remove first column (row count)
trn_ds2<-as.data.table(subset(trn_ds2, select =ds_names2[2:rem_colct]))
rem_colct<-rem_colct - 1  #removed first column
trn_ds2<-as.data.table(trn_ds2)

#partitions the data into 2 sets (60/40 split)
trn_ds2 <- trn_ds2[, classe := factor(trn_ds2[, classe])]
trn_ds2[, .N, classe]
```

```
##    classe    N
## 1:      A 5580
## 2:      B 3797
## 3:      C 3422
## 4:      D 3216
## 5:      E 3607
```

```r
trn_part <- createDataPartition(y=trn_ds2$classe, p=0.6, list=FALSE)
First_trn_ds<-as.data.frame(trn_ds2)[trn_part,]
Second_trn_ds <- as.data.frame(trn_ds2)[-trn_part,]
names(First_trn_ds)=names(trn_ds2)
names(Second_trn_ds)=names(trn_ds2)

# displays # of observations and # of variables for first set (60% set)
dim(First_trn_ds)
```

```
## [1] 11776    58
```

```r
# displays # of observations and # of variables for second set (40% set)
dim(Second_trn_ds)
```

```
## [1] 7846   58
```

##Model Selection and Building the Model

Once the data has been cleaned and subsetted, two different models were applied and a comparison made between the two.  The first model applied was a random forrest model,   Below is the code used in generating the model and the output of the model statistics:


```r
# fit model using random forest
modfit <- train(classe ~ ., method = "rf", data = First_trn_ds , importance = TRUE, trControl = trainControl(method = "cv", number = 10))
print(modfit)
```

```
## Random Forest 
## 
## 11776 samples
##    57 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## 
## Summary of sample sizes: 10599, 10598, 10598, 10597, 10599, 10599, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD   Kappa SD    
##    2    0.9872620  0.9838843  0.0033746787  0.0042671993
##   40    0.9990659  0.9988186  0.0009344771  0.0011819447
##   79    0.9984714  0.9980666  0.0007807454  0.0009875255
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 40.
```
The model was then run against the second data set and its accuracy was checked using the confusionMatrix function.


```r
predict1 <- predict(modfit, Second_trn_ds, type = "raw")
confusionMatrix(predict1, Second_trn_ds$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2232    3    0    0    0
##          B    0 1512    2    0    0
##          C    0    3 1366    0    0
##          D    0    0    0 1286    0
##          E    0    0    0    0 1442
## 
## Overall Statistics
##                                          
##                Accuracy : 0.999          
##                  95% CI : (0.998, 0.9996)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9987         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9960   0.9985   1.0000   1.0000
## Specificity            0.9995   0.9997   0.9995   1.0000   1.0000
## Pos Pred Value         0.9987   0.9987   0.9978   1.0000   1.0000
## Neg Pred Value         1.0000   0.9991   0.9997   1.0000   1.0000
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1927   0.1741   0.1639   0.1838
## Detection Prevalence   0.2849   0.1930   0.1745   0.1639   0.1838
## Balanced Accuracy      0.9997   0.9979   0.9990   1.0000   1.0000
```

##
The second model generated was a boosted tree model using the gbm package.  Its accuracy was also checked using the confusionMatrix function.  below is the code and the output:


```r
# this code bit was retrieved from http://topepo.github.io/caret/training.html
# just sets control feture of the gbm method into a variable
fitControl <- trainControl(## 10-fold CV
  method = "repeatedcv",
  number = 10,
  ## repeated ten times
  repeats = 0)

#fit a boosted tree model via the gbm package.
gbmFit1 <- train(classe ~ ., data = First_trn_ds,
                 method = "gbm",
                 trControl = fitControl,
                 verbose = FALSE)
print(gbmFit1)
```

```
## Stochastic Gradient Boosting 
## 
## 11776 samples
##    57 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold, repeated 0 times) 
## 
## Summary of sample sizes: 10599, 10599, 10599, 10598, 10599, 10599, ... 
## 
## Resampling results across tuning parameters:
## 
##   interaction.depth  n.trees  Accuracy   Kappa      Accuracy SD
##   1                   50      0.8398014  0.7967684  0.012047027
##   1                  100      0.9003487  0.8738515  0.007996658
##   1                  150      0.9278619  0.9086332  0.008167081
##   2                   50      0.9560123  0.9443156  0.006676089
##   2                  100      0.9859464  0.9822237  0.003422760
##   2                  150      0.9918479  0.9896892  0.002677121
##   3                   50      0.9818282  0.9770102  0.004424513
##   3                  100      0.9933766  0.9916226  0.001754487
##   3                  150      0.9956694  0.9945224  0.001865958
##   Kappa SD   
##   0.015274698
##   0.010115685
##   0.010347640
##   0.008452097
##   0.004328528
##   0.003384663
##   0.005598926
##   0.002218488
##   0.002360220
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## Accuracy was used to select the optimal model using  the largest value.
## The final values used for the model were n.trees = 150,
##  interaction.depth = 3 and shrinkage = 0.1.
```
and checking the model against the test set:

```r
predict2 <- predict(gbmFit1, Second_trn_ds, type = "raw")
confusionMatrix(predict2, Second_trn_ds$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2232    3    0    0    0
##          B    0 1511    2    0    0
##          C    0    4 1359    3    0
##          D    0    0    7 1276    4
##          E    0    0    0    7 1438
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9962          
##                  95% CI : (0.9945, 0.9974)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9952          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9954   0.9934   0.9922   0.9972
## Specificity            0.9995   0.9997   0.9989   0.9983   0.9989
## Pos Pred Value         0.9987   0.9987   0.9949   0.9915   0.9952
## Neg Pred Value         1.0000   0.9989   0.9986   0.9985   0.9994
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1926   0.1732   0.1626   0.1833
## Detection Prevalence   0.2849   0.1928   0.1741   0.1640   0.1842
## Balanced Accuracy      0.9997   0.9975   0.9962   0.9953   0.9981
```
##Conclusion
By the model output summaries and the confusion matrices on the test set, the random forest model yielded marginally better results than the boosted tree model, however the differences are so small they could be a result of random occurance.  Both the forest model and the boosted tree model yielded acceptable results.  

