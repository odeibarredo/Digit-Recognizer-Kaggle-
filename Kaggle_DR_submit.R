#### KAGGLE - DIGIT RECOGNITION ####

### LIBRARY NEEDED:
library(rpart)
library(e1071)
library(randomForest)
library(caret)

### IMPORT DATA:
original_train <- read.csv("train.csv", header = T, sep = ",")
original_test <- read.csv("test.csv", header = T, sep = ",")

### DATA MINING:
original_train$label <- as.factor(original_train$label)
table(original_train$label)

# Digits appear in similar amount in the train dataset, which is good!
# We can visualize an image to see how the digits appear
# Let's check a random image, for example the one in row 7

# Extract the corresponding data
row7 <- as.integer(original_train[7, -1])

# Create a matrix with dimensions of 28x28 to create the image (as it 
# is described in the description of the dataset)
mat7 <- t(matrix(row7, 28, 28))

# Visualize the image using a color gradient from 0 to 255 (minimum and
# maximum values for the pixels)
image(mat7, col=grey.colors(255))

# We need to rotate the image
rotate <- function(x) t(apply(x, 2, rev))
mat7_rot <- rotate(mat7)
image(mat7_rot, col=grey.colors(255))

# Ironically the digit of the seventh row is 7


### DATA TREATMENT
# First thing that comes to mind is to remove all the pixels that
# have a value of "0" in all images
reduced <- original_train[, colSums(original_train != 0) > 0]

# From 784 pixels to 708, not much...
# We must remove other pixels with little variance through images
# *pixels with little variances will be the ones with most 0 in them

variances <- data.frame(apply(reduced[-1], 2, var))
colnames(variances) <- "variances"

plot(variances$variances, type = "l", xlab="Pixel",
     ylab="Pixel variance", lwd=2)

sorted_var <- variances[order(variances$variances), , drop = FALSE]
# "drop = False" allows us to maintain original row names

plot(sorted_var$variances, type = "l",xlab="Pixel",
     ylab="Pixel variance", lwd=2)
# there are about 200 pixels with very low variance
par(mar=c(3,9,3,9))
boxplot(variances, ylab="Pixel variance")
abline(h=4850, col="red", lwd=2) # mean value
text(5300, "Mean", font=2)
abline(h=200, col="blue", lwd=2) # where are those 200 pixels
text(600, "200 pixels", font=2)


# Seems like the first quantile includes near 200 pixels
# let's confirm that by using the 1st Qu. values from the summary
summary(variances) # 1st Qu. = 89
length(variances[variances$variances <= 89, ]) # 177 pixels, close

# Let's zoom in into the previous plot to see where does 200 pixels 
# exactly cut
plot(sorted_var$variances, type = "l", xlim = c(0,220), ylim=c(0,200))
abline(v=200) # around 150
length(variances[variances$variances <= 150, ]) # exactly 200 pixels

# Remove those pixels
less_200 <- subset(variances, variances >= 150, "variances")
pixels <- row.names(less_200)
train <- original_train[, c("label", pixels)] 


### MACHINE LEARNING
# First we normalize the data
normalize <- function(x) {
  num <- x - min(x)
  denom <- max(x) - min(x)
  return (num/denom)
}

train_nolabel <- train[,-1]
train_nolabel_norm <- as.data.frame(lapply(train_nolabel, normalize))
train_norm <- cbind(label=train$label, train_nolabel_norm)

# Split training data
indexes <- sample(1:nrow(train_norm), size=0.8*nrow(train_norm))
t.train <- train_norm[indexes, ]
t.test <- train_norm[-indexes, ]

# One of the most common prediction models is the decision tree
# Let's try it out
dt_model <- rpart(label ~ ., t.train, method = "class")
dt_pred <- predict(dt_model, t.test, type = "class")
dt_pred

confusionMatrix(t.test$label, dt_pred)
# (~60% accuracy is pretty poor...)

# Naive Bayes:
nb_model <- naiveBayes(label ~ ., t.train, method = "class")
nb_pred <- predict(dt_model, t.test, type = "class")
nb_pred

confusionMatrix(t.test$label, nb_pred)
# (Same results as decision tree)

# Random Forest:
# Training a random forest model takes a long time due to the
# data volume, so in this case I only use 2000 images. 
# It serves to see how well it does
rf_model <- randomForest(label ~ ., t.train[1:2000,])
rf_pred <- predict(rf_model, t.test)
rf_pred

confusionMatrix(t.test$label, rf_pred)
# (~%92, that's a huge increase in the accuracy!)

# SVM:
# This model also takes a lot of time, so I do the same, just to
# see what happens
svm_model <- train(label~.,t.train[1:2000,], method="svmRadial")
svm_pred <- predict(svm_model, t.test)
svm_pred

confusionMatrix(t.test$label, svm_pred)
# (SVM also achieves really good results, but it takes longer
# to train the model)

# So, in recap, we've seen that Random Forest and SVM are the
# best methods to make our predictions. Since RF is faster we will
# use that.


### PREDICITIONS:
# In order to use our model to predict the test dataset, we need
# to pre-process the test data the same way we did with the train data

# First we remove the pixels we didn't use
reduced_test <- original_test[,pixels]
# Then we normalize
test_norm <- as.data.frame(lapply(reduced_test, normalize))

# Make the predictions
pred <- predict(rf_model, test_norm)
table(pred)

pred_submit <- data.frame(ImageId=1:nrow(original_test),Label=pred)

write.table(pred_submit, "DigRec_submit_1.csv", sep=";",quote = F, row.names = F)

### FINAL NOTE:
# To obtain better results just train the model with greater amount
# of data. Also you can play with the amount of pixels to remove from
# the dataset, based on the variance.

