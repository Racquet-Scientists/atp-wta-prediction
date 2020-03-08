##############################
#####  PACKAGE SETUP     #####
##############################
#List the packages we need, install if missing, then load all of them
PackageList =c('tidyverse','tree','rpart','rpart.plot','randomForest','gbm','kknn','glmnet', 'miceadds')
NewPackages=PackageList[!(PackageList %in%
                            installed.packages()[,"Package"])]
if(length(NewPackages)) install.packages(NewPackages)
lapply(PackageList,require,character.only=TRUE)

####     Functions     ######
#############################
# Deviance
loss_function = function(y,phat,correction=0.00001) {
  if(is.factor(y)) y = as.numeric(y)-1 
  phat = (1-correction)*phat + correction*.5
  py = ifelse(y==1, phat, 1-phat) 
  return(-2*sum(log(py)))
}

# Set Seed
set.seed(6992) 

####      SETUP        ######
#############################
# Data is located in Results.Rdata
load.Rdata(filename="Results.Rdata", objname = "tennis_data") 
           
# Change variables to factors / numerical values where applicable
summary(tennis_data)

# Partition Data to Train/Test/Validation
# Use 60% for Train (to be used for models)
# Use 20% for Validation (to evaluate models with Deviance loss)
# Use 20% for Test (once model is selected, report Accuracy with Test set)
n = 0.8
sample_size = floor(n*nrow(tennis_data))
train_validation_index = sample(seq_len(nrow(tennis_data)),size = sample_size)
train_validation = tennis_data[train_validation_index,]
test_set = tennis_data[-train_validation_index,]
n = 0.75
sample_size = floor(n*nrow(train_validation))
train_index = sample(seq_len(nrow(train_validation)),size = sample_size)
train_set = train_validation[train_index,]
validation_set = train_validation[-train_index,]

# Create data frames to be used (if not in data frame yet)

# Use threshold = 0.5 for all models
threshold = 0.5
p_hat_L = list() # list of probabilities calculated by each model; to be used for deviance calculation
y_hat_L = list() # list of predictions calculated by each model

#### Logistic Regression #####
##############################
# Run LASSO to evaluate variable selection
# Evaluate LASSO with min lambda and lambda 1 standard error

# Run GLM & Inspect
# change names for variables (x's and y) and data set used
lr_fit = glm(y~x, train, family=binomial(link = "logit"))
p_hat_lr = predict(lr_fit, validation, type="response")

# Store probabilities
p_hat_L$LR = matrix(p_hat_lr,ncol = 1)

# Store predictions (based on probability and threshold value)
y_hat_L$LR = p_hat_lr
y_hat_L$LR[p_hat_lr >= threshold] = 1
y_hat_L$LR[p_hat_lr < threshold] = 0

#### K Nearest Neighbor #####
#############################
# Normalize data (or better yet standardized with mean = 0 and sd = 1)
# x_standardized = scale(x_not_standardized)
# Run KNN using CV or maybe Caret internal CV to find optimal K
# Choose values for n with "from" and "to"
from = 2
to = 30
n = to - from + 1
kk = seq(from, to,((to - from)/(n - 1)))
p_hat_L$kNN = matrix(0.0,nrow = nrow(data_set), ncol = n)
y_hat_L$kNN = matrix(0.0,nrow = nrow(data_set), ncol = n)
for(i in kk) {
  cat("on k = ",i,"\n")
  # change names for variables (x's and y) and data set used
  kk_fit = kknn(y~x,
                train=train,
                test=validation,
                k=i,kernel = "rectangular")
# Store probabilities
  phat = as.numeric(kk_fit$fitted.values)-1
  p_hat_L$kNN[,i-1] = phat
# Store predictions (based on probability and threshold value)
  y_hat_L$kNN[,i-1] = phat
  y_hat_L[phat >= threshold] = 1
  y_hat_L[phat < threshold] = 0
}

#### Classification Tree #####
##############################
# Build Complex Tree
# control=rpart.control(minsplit = 5,cp=0.00025,xval=10))
tree_fit = rpart(y ~ x, data = train,
                 method = "class",
                 control=rpart.control(
                   minsplit = 5,
                   cp=0.00025,
                   xval=10))

# Prune Complex Tree based on best cp value considering standard error
# Selecting cp value considering statistical significance
cp_best = tree_fit$cptable[which.min(tree_fit$cptable[,"xerror"]),"CP"]
index_cp_min = which.min(tree_fit$cptable[,"xerror"])
val_h = tree_fit$cptable[index_cp_min, "xerror"] + tree_fit$cptable[index_cp_min, "xstd"]
index_cp_std = Position(function(x) x < val_h, tree_fit$cptable[, "xerror"])
cp_std = tree_fit$cptable[index_cp_std,"CP"]
# cross-validation results - use for report
#par(mfrow=c(1,1))
#plotcp(tree_fit)
# Best cp considering statistical significance level
cp_best = cp_std
# Pruning tree
tree_fit = prune(tree_fit,cp_best)
# visualizing model - for report
#par(mfrow=c(1,1))
#rpart.plot(tree_fit, main="Pruned Regression Tree")

# Store probabilities
p_hat = predict(object = tree_fit, newdata = validation, type = "prob")
p_hat_L$CART = p_hat

# Store predictions (based on probability and threshold value)
y_hat_L$CART = phat
y_hat_L[phat >= threshold] = 1
y_hat_L[phat < threshold] = 0

#### Random Forest #####
########################
# Build RF with different paramter values
p = dim(data_set)[2]-1
mtryv = c(p,sqrt(p)) #number of variables
ntreev = c(50,100,250,500) #number of trees
setrf = expand.grid(mtryv,ntreev) 
colnames(setrf)=c("mtry","ntree") 
p_hat_L$RF = matrix(0.0,nrow = nrow(validation), ncol = nrow(setrf))
y_hat_L$RF = matrix(0.0,nrow = nrow(validation), ncol = nrow(setrf))

for(i in 1:nrow(setrf)) {
  cat("on randomForest fit ",i,", mtry=",setrf[i,1],", B=",setrf[i,2],"\n")
  fit_rf = randomForest(y ~ x,
                        data=train,
                        mtry=setrf[i,1],
                        ntree=setrf[i,2]) 
  p_hat = predict(fit_rf,validation,type="prob")

# Store probabilities - all models
  p_hat_L$RF[,i] = p_hat

# Store predictions (based on probability and threshold value) - all models
  y_hat_L$RF[,i] = p_hat
  y_hat_L$RF[p_hat >= threshold, i] = 1
  y_hat_L$RF[p_hat < threshold, i] = 0
}
  
####    Boosting   #####
########################
# Build Boosting tree with different paramter values
idv = c(2,4) #Depth
ntv = c(1000,5000) #Number of trees
shv = c(0.1,0.01) #Shrinking
setboost = expand.grid(idv,ntv,shv)
colnames(setboost) = c("tdepth","ntree","shrink")
p_hat_L$Boost = matrix(0.0,nrow(validation),nrow(setboost))
y_hat_L$Boost = matrix(0.0,nrow(validation),nrow(setboost))

for(i in 1:nrow(setboost)) {
  cat("on boosting fit",i,
      ", idv = ",setboost[i,1],
      ", ntv = ",setboost[i,2],
      ", shv = ",setboost[i,3],"\n")
  fboost = gbm(y~x, data=train, distribution="bernoulli",
               n.trees=setboost[i,2], 
               interaction.depth=setboost[i,1], 
               shrinkage=setboost[i,3])
  p_hat = predict(fboost, newdata=validation,n.trees=setboost[i,2], type="response")

# Store probabilities - all models
  p_hat_L$Boost[,i] = p_hat

# Store predictions (based on probability and threshold value) - all models
  y_hat_L$Boost[,i] = p_hat
  y_hat_L$Boost[p_hat >= threshold, i] = 1
  y_hat_L$Boost[p_hat < threshold, i] = 0
}

#### Aggreggate Methods  #####
##############################
# Probably out of scope; 
# we would have to use all models considered for the aggreggate method to get the predictions
# Use Average & Majority Votes
# We could average all probabilities, and get outcome by applying threshold
# How would we get the probability using majority vote? Averaging probabilities for those in favor of the outcome?

# Store probabilities

# Store predictions (using threshold on probabilities)


####   Model Evaluation  #####
##############################
# Calculate Deviance for all models (using probabilities and predictions)
loss_L = list()
nmethod = length(p_hat_L) 
# Set y to outcomes in data set
y = as.numeric(validation$y)-1 # modify to match data set and variable name
for(i in 1:nmethod) {
  # Run for all methods
  nrun = ncol(p_hat_L[[i]])
  lvec = rep(0,nrun)
  for(j in 1:nrun) {
    # Run for variations (different tuning parameters) within method
    lvec[j] = loss_function(y, p_hat_L[[i]][,j])
  }
  loss_L[[i]]=lvec
  names(loss_L)[i] = names(p_hat_L)[i]
}
# Visualizing results
lossv = unlist(loss_L)
par(mfrow=c(1,1))
plot(lossv, ylab="Loss on Validation Set", type="n") 
nloss=0 # used as offset for loop
all_losses = NULL
for(i in 1:nmethod) {
  # Calculate indexes to use
  j = nloss + 1:ncol(p_hat_L[[i]]) 
  all_losses = c(all_losses,lossv[j])
  points(j,lossv[j],col=i,pch=17) 
  # Update offset
  nloss = nloss + ncol(p_hat_L[[i]])
} 
legend("topright",legend=names(p_hat_L),col=1:nmethod,pch=rep(17,nmethod),cex = 0.9)

# Show confusion matrix for best model within each group
# Logistic Regression
table(predictions = y_hat_L$LR, actual = y)

# kNN
i = 1 # choose best model
table(predictions = y_hat_L$kNN[,i], actual = y)

# CART
table(predictions = y_hat_L$CART, actual = y)

# Random Forest
i = 1 # choose best model
table(predictions = y_hat_L$RF[,i], actual = y)

# Boosting
i = 1 # choose best model
table(predictions = y_hat_L$Boosting[,i], actual = y)

# Aggregate method

# See which model is best (minimum deviance)
which.min(all_losses)

# Retrain best model with train + validation

# Use test set to give a final accuracy description of the model


#### Torunament Prediction #####
################################
# Retrain best model with entire data set (train + validation + test)

# Use tournament data (to be provided by Carly / Sylvia)
# Predict brackets and create following rounds iteratively

# Summarize French Open results with each brakcet winners up to final winner
