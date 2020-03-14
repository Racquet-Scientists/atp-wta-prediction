##############################
#####  PACKAGE SETUP     #####
##############################
#List the packages we need, install if missing, then load all of them
PackageList =c('tidyverse','tree','rpart','rpart.plot','randomForest','gbm','kknn','glmnet', 'miceadds','ROCR')
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
  return (-2*sum(log(py)))
}
# Accuracy for a 2x2 confusion matrix
accuracy_from_cm = function(cm) {
  return ((cm[1] + cm[4])/sum(cm))
}

# Set Seed
set.seed(6992) 

####      SETUP        ######
#############################
# Data is located in Results.Rdata
load.Rdata(filename="Results.Rdata", objname = "tennis_data") 
feature_selection = FALSE
all_variables = FALSE
variable_importance = FALSE

# Change variables to factors / numerical values where applicable
summary(tennis_data)
names(tennis_data)[6] = "BestOf"
# Factors
tennis_data$Tournament = as.factor(tennis_data$Tournament)
tennis_data$Year = as.factor(tennis_data$Year)
tennis_data$Court = as.factor(tennis_data$Court)
tennis_data$Surface = as.factor(tennis_data$Surface)
tennis_data$Round = as.factor(tennis_data$Round)
tennis_data$BestOf = as.factor(tennis_data$BestOf)
tennis_data$Player1 = as.factor(tennis_data$Player1)
tennis_data$Player2 = as.factor(tennis_data$Player2)
tennis_data$Outcome = as.factor(tennis_data$Outcome)
# Numeric
tennis_data$P1Rank = as.numeric(tennis_data$P1Rank)
#na_index = is.na(tennis_data$P1Rank)
#tennis_data$P1Rank[na_index] = max(tennis_data$P1Rank, na.rm=TRUE) + 1
tennis_data$P2Rank = as.numeric(tennis_data$P2Rank)
#na_index = is.na(tennis_data$P2Rank)
#tennis_data$P2Rank[na_index] = max(tennis_data$P2Rank, na.rm=TRUE) + 1
tennis_data$P1Pts = as.numeric(tennis_data$P1Pts)
#na_index = is.na(tennis_data$P1Pts)
#tennis_data$P1Pts[na_index] = 0
tennis_data$P2Pts = as.numeric(tennis_data$P2Pts)
#na_index = is.na(tennis_data$P2Pts)
#tennis_data$P2Pts[na_index] = 0
# All other values are already numeric

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
train_set = as.data.frame(train_set)
validation_set = as.data.frame(validation_set)
test_set = as.data.frame(test_set)
train_validation_set = as.data.frame(train_validation)
tennis_data_set = as.data.frame(tennis_data)

# Use threshold = 0.5 for all models
threshold = 0.5
p_hat_L = list() # list of probabilities calculated by each model; to be used for deviance calculation
y_hat_L = list() # list of predictions calculated by each model

####  Feature Selection  #####
##############################
# Run LASSO to evaluate variable selection
if (feature_selection) {
  x = model.matrix(Outcome~.,train_set)
  y = train_set$Outcome
  lasso_fit = cv.glmnet(x,y,alpha=1,family="binomial",type.measure="mse")
  # Evaluate LASSO with min lambda and lambda 1 standard error
  # Visual representation
  par(mfrow=c(1,1))
  plot(lasso_fit)
  lambda_min = lasso_fit$lambda.min
  lambda_1se = lasso_fit$lambda.1se
  log(lambda_min)
  log(lambda_1se)
  # Redirecting output to file (too large to see on terminal)
  options(max.print= 1000000, width = 1000)
  sink("variable_selection", append=FALSE, split=FALSE)
  coef(lasso_fit,s=lambda_min)
  # Variables showing coefficient values (selected by LASSO):
  # Player1, Player2, P1Pts, P2Pts, Player1Srv1Wp, Player1Srv2Wp, Player1GamesWp, Player1MatchesWp, Player1SetWp, Player2Srv1p, Player2Srv1Wp, Player2GamesWp, Player2MatchesWp, Player2SetWp
  coef(lasso_fit,s=lambda_1se)
  # Variables showing coefficient values (selected by LASSO):
  # Player1, Player2, P1Pts, P2Pts, Player1Srv1Wp, Player1GamesWp, Player1MatchesWp, Player1SetWp, Player2Srv1Wp, Player2GamesWp, Player2MatchesWp, Player2SetWp
  
  # return to default output
  sink()
  closeAllConnections()
  options(max.print= 99999, width = 80) # Back to defaults
}

# Feature selection:
# Player1, Player2 will be omitted given randomness of samples
# P1Pts, P2Pts 
# Player1Srv1Wp, Player1GamesWp, Player1MatchesWp, Player1SetWp
# Player2Srv1Wp, Player2GamesWp, Player2MatchesWp, Player2SetWp
# + Outcome
# Redefining data frames with selected variables
if (all_variables) {
  train_set = as.data.frame(train_set %>%
                              select(Outcome,Year,Court,Surface,Round,BestOf,P1Rank,P2Rank,P1Pts,P2Pts,
                                     Player1ACp,Player1DFp,Player1Srv1p,Player1Srv1Wp,Player1Srv2Wp,Player1Srv1ReWp,Player1Srv2ReWp,
                                     Player1GamesWp,Player1MatchesWp,Player1SetWp,
                                     Player2ACp,Player2DFp,Player2Srv1p,Player2Srv1Wp,Player2Srv2Wp,Player2Srv1ReWp,Player2Srv2ReWp,
                                     Player2GamesWp,Player2MatchesWp,Player2SetWp,))
  validation_set = as.data.frame(validation_set %>%
                                   select(Outcome,Year,Court,Surface,Round,BestOf,P1Rank,P2Rank,P1Pts,P2Pts,
                                          Player1ACp,Player1DFp,Player1Srv1p,Player1Srv1Wp,Player1Srv2Wp,Player1Srv1ReWp,Player1Srv2ReWp,
                                          Player1GamesWp,Player1MatchesWp,Player1SetWp,
                                          Player2ACp,Player2DFp,Player2Srv1p,Player2Srv1Wp,Player2Srv2Wp,Player2Srv1ReWp,Player2Srv2ReWp,
                                          Player2GamesWp,Player2MatchesWp,Player2SetWp,))
  test_set = as.data.frame(test_set %>%
                             select(Outcome,Year,Court,Surface,Round,BestOf,P1Rank,P2Rank,P1Pts,P2Pts,
                                    Player1ACp,Player1DFp,Player1Srv1p,Player1Srv1Wp,Player1Srv2Wp,Player1Srv1ReWp,Player1Srv2ReWp,
                                    Player1GamesWp,Player1MatchesWp,Player1SetWp,
                                    Player2ACp,Player2DFp,Player2Srv1p,Player2Srv1Wp,Player2Srv2Wp,Player2Srv1ReWp,Player2Srv2ReWp,
                                    Player2GamesWp,Player2MatchesWp,Player2SetWp,))
  train_validation_set = as.data.frame(train_validation_set %>%
                                         select(Outcome,Year,Court,Surface,Round,BestOf,P1Rank,P2Rank,P1Pts,P2Pts,
                                                Player1ACp,Player1DFp,Player1Srv1p,Player1Srv1Wp,Player1Srv2Wp,Player1Srv1ReWp,Player1Srv2ReWp,
                                                Player1GamesWp,Player1MatchesWp,Player1SetWp,
                                                Player2ACp,Player2DFp,Player2Srv1p,Player2Srv1Wp,Player2Srv2Wp,Player2Srv1ReWp,Player2Srv2ReWp,
                                                Player2GamesWp,Player2MatchesWp,Player2SetWp,))
  tennis_data_set = as.data.frame(tennis_data_set %>%
                                    select(Outcome,Year,Court,Surface,Round,BestOf,P1Rank,P2Rank,P1Pts,P2Pts,
                                           Player1ACp,Player1DFp,Player1Srv1p,Player1Srv1Wp,Player1Srv2Wp,Player1Srv1ReWp,Player1Srv2ReWp,
                                           Player1GamesWp,Player1MatchesWp,Player1SetWp,
                                           Player2ACp,Player2DFp,Player2Srv1p,Player2Srv1Wp,Player2Srv2Wp,Player2Srv1ReWp,Player2Srv2ReWp,
                                           Player2GamesWp,Player2MatchesWp,Player2SetWp,))
                                    
} else {
  train_set = as.data.frame(train_set %>%
                              select(Outcome,P1Pts,P2Pts,
                                     Player1Srv1Wp,Player1GamesWp,Player1MatchesWp,Player1SetWp,
                                     Player2Srv1Wp,Player2GamesWp,Player2MatchesWp,Player2SetWp))
  validation_set = as.data.frame(validation_set %>%
                                   select(Outcome,P1Pts,P2Pts,
                                          Player1Srv1Wp,Player1GamesWp,Player1MatchesWp,Player1SetWp,
                                          Player2Srv1Wp,Player2GamesWp,Player2MatchesWp,Player2SetWp))
  test_set = as.data.frame(test_set %>%
                             select(Outcome,P1Pts,P2Pts, 
                                    Player1Srv1Wp,Player1GamesWp,Player1MatchesWp,Player1SetWp,
                                    Player2Srv1Wp,Player2GamesWp,Player2MatchesWp,Player2SetWp))
  train_validation_set = as.data.frame(train_validation_set %>%
                                         select(Outcome,P1Pts,P2Pts, 
                                                Player1Srv1Wp,Player1GamesWp,Player1MatchesWp,Player1SetWp,
                                                Player2Srv1Wp,Player2GamesWp,Player2MatchesWp,Player2SetWp))
  tennis_data_set = as.data.frame(tennis_data_set %>%
                                    select(Outcome,P1Pts,P2Pts, 
                                           Player1Srv1Wp,Player1GamesWp,Player1MatchesWp,Player1SetWp,
                                           Player2Srv1Wp,Player2GamesWp,Player2MatchesWp,Player2SetWp))
}

#### Logistic Regression #####
##############################
# Run GLM & Inspect
# change names for variables (x's and y) and data set used
lr_fit = glm(Outcome~., train_set, family=binomial(link = "logit"))
p_hat_lr = predict(lr_fit, validation_set, type="response")

# Store probabilities
p_hat_L$LR = matrix(p_hat_lr,ncol = 1)

# Store predictions (based on probability and threshold value)
y_hat_L$LR = matrix(p_hat_lr,ncol = 1)
y_hat_L$LR[p_hat_lr >= threshold] = 1
y_hat_L$LR[p_hat_lr < threshold] = 0

#### K Nearest Neighbor #####
#############################
# Normalize data (or better yet standardized with mean = 0 and sd = 1)
varnames = names(train_set)[-1]
if (all_variables) {
  varnames = varnames[-(1:5)]
}
index = names(train_set) %in% varnames
temp = scale(train_set[, index])
train_set_normalized = train_set
train_set_normalized[, index] = temp
temp = scale(validation_set[, index])
validation_set_normalized = validation_set
validation_set_normalized[, index] = temp
temp = scale(test_set[, index])
test_set_normalized = test_set
test_set_normalized[, index] = temp

# Choose values for n with "from" and "to"
from = 2
to = 20
n = to - from + 1
kk = seq(from, to,((to - from)/(n - 1)))
p_hat_L$kNN = matrix(0.0,nrow = nrow(validation_set_normalized), ncol = n)
y_hat_L$kNN = matrix(0.0,nrow = nrow(validation_set_normalized), ncol = n)
for(i in kk) {
  cat("on k = ",i,"\n")
  # change names for variables (x's and y) and data set used
  kk_fit = kknn(Outcome~.,
                train=train_set_normalized,
                test=validation_set_normalized,
                k=i,kernel = "rectangular")
# Store probabilities
  phat = kk_fit$prob[,2]
  p_hat_L$kNN[,i-1] = phat
# Store predictions (based on probability and threshold value)
  y_hat_L$kNN[,i-1] = phat
  y_hat_L$kNN[phat >= threshold, i-1] = 1
  y_hat_L$kNN[phat < threshold, i-1] = 0
}

#### Classification Tree #####
##############################
# Build Complex Tree
tree_fit = rpart(Outcome ~ ., data = train_set,
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
# cross-validation results
par(mfrow=c(1,1))
plotcp(tree_fit)
# Best cp considering statistical significance level
cp_best = cp_std
# Pruning tree
tree_fit = prune(tree_fit,cp_best)
# visualizing model
par(mfrow=c(1,1))
rpart.plot(tree_fit, main="Pruned CART")

# Store probabilities
p_hat = predict(object = tree_fit, newdata = validation_set, type = "prob")
p_hat_L$CART = matrix(p_hat[,2],ncol = 1)

# Store predictions (based on probability and threshold value)
y_hat_L$CART = matrix(p_hat[,2],ncol = 1)
y_hat_L$CART[p_hat[,2] >= threshold] = 1
y_hat_L$CART[p_hat[,2] < threshold] = 0

#### Random Forest #####
########################
# Test variable importance
if (variable_importance) {
  rf_test = randomForest(Outcome~.,
                         data=train_set, #data set
                         mtry=sqrt(dim(train_set)[2]-1), #number of variables to sample
                         ntree=500, #number of trees to grow
                         nodesize=1,#minimum node size on trees (optional) 
                         maxnodes=40, #maximum number of terminal nodes (optional)
                         importance=TRUE, #calculate variable importance measure (optional)
  )
  num_of_variables = 10
  if (all_variables) {
    num_of_variables = 15
  }
  varImpPlot(rf_test, sort = TRUE, num_of_variables)
}
# Build RF with different paramter values
p = dim(train_set)[2]-1 # We subtract one to account for Outcome
mtryv = c(p,sqrt(p)) #number of variables
ntreev = c(50,100,250,500) #number of trees
setrf = expand.grid(mtryv,ntreev) 
colnames(setrf)=c("mtry","ntree") 
p_hat_L$RF = matrix(0.0,nrow = nrow(validation_set), ncol = nrow(setrf))
y_hat_L$RF = matrix(0.0,nrow = nrow(validation_set), ncol = nrow(setrf))

for(i in 1:nrow(setrf)) {
  cat("on randomForest fit ",i,", mtry=",setrf[i,1],", B=",setrf[i,2],"\n")
  fit_rf = randomForest(Outcome ~ .,
                        data=train_set,
                        mtry=setrf[i,1],
                        ntree=setrf[i,2]) 
  p_hat = predict(fit_rf,validation_set,type="prob")

# Store probabilities - all models
  p_hat_L$RF[,i] = p_hat[,2]

# Store predictions (based on probability and threshold value) - all models
  y_hat_L$RF[,i] = p_hat[,2]
  y_hat_L$RF[p_hat[,2] >= threshold, i] = 1
  y_hat_L$RF[p_hat[,2] < threshold, i] = 0
}
  
####    Boosting   #####
########################
# Change outcome to numerical
train_set_numerical = train_set
train_set_numerical$Outcome = as.numeric(train_set$Outcome)-1
validation_set_numerical = validation_set
validation_set_numerical$Outcome = as.numeric(validation_set$Outcome)-1
test_set_numerical = test_set
test_set_numerical$Outcome = as.numeric(test_set$Outcome)-1
train_validation_set_numerical = train_validation_set
train_validation_set_numerical$Outcome = as.numeric(train_validation_set$Outcome)-1
tennis_data_set_numerical = tennis_data_set
tennis_data_set_numerical$Outcome = as.numeric(tennis_data_set$Outcome)-1
# Build Boosting tree with different paramter values
idv = c(2,4) #Depth
ntv = c(1000,5000) #Number of trees
shv = c(0.1,0.01) #Shrinking
setboost = expand.grid(idv,ntv,shv)
colnames(setboost) = c("tdepth","ntree","shrink")
p_hat_L$Boost = matrix(0.0,nrow(validation_set_numerical),nrow(setboost))
y_hat_L$Boost = matrix(0.0,nrow(validation_set_numerical),nrow(setboost))

for(i in 1:nrow(setboost)) {
  cat("on boosting fit",i,
      ", idv = ",setboost[i,1],
      ", ntv = ",setboost[i,2],
      ", shv = ",setboost[i,3],"\n")
  fboost = gbm(Outcome~., data=train_set_numerical, distribution="bernoulli",
               n.trees=setboost[i,2], 
               interaction.depth=setboost[i,1], 
               shrinkage=setboost[i,3])
  p_hat = predict(fboost, newdata=validation_set_numerical,n.trees=setboost[i,2], type="response")

# Store probabilities - all models
  p_hat_L$Boost[,i] = p_hat

# Store predictions (based on probability and threshold value) - all models
  y_hat_L$Boost[,i] = p_hat
  y_hat_L$Boost[p_hat >= threshold, i] = 1
  y_hat_L$Boost[p_hat < threshold, i] = 0
}

####   Model Evaluation  #####
##############################
# Calculate Deviance for all models (using probabilities and predictions)
loss_L = list()
nmethod = length(p_hat_L) 
# Set y to outcomes in data set
y = as.numeric(validation_set$Outcome)-1
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
cm = table(predictions = y_hat_L$LR, actual = y)
accuracy_lr = accuracy_from_cm(cm)
print("Logistic Regression:")
print(cm)
print(accuracy_lr)

# kNN
i = 19 # choose best model
cm = table(predictions = y_hat_L$kNN[,i], actual = y)
accuracy_kNN = accuracy_from_cm(cm)
print("k Nearest Neighbor:")
print(cm)
print(accuracy_kNN)

# CART
cm = table(predictions = y_hat_L$CART, actual = y)
accuracy_CART = accuracy_from_cm(cm)
print("CART:")
print(cm)
print(accuracy_CART)

# Random Forest
i = 8 # choose best model
cm = table(predictions = y_hat_L$RF[,i], actual = y)
accuracy_RF = accuracy_from_cm(cm)
print("Random Forest:")
print(cm)
print(accuracy_RF)

# Boosting
i = 6 # choose best model
cm = cm = table(predictions = y_hat_L$Boost[,i], actual = y)
accuracy_Boost = accuracy_from_cm(cm)
print("Boosting:")
print(cm)
print(accuracy_Boost)

####     ROC Curves      #####
##############################
p_hat_best_models = list()
p_hat_best_models$LR = p_hat_L$LR
i = 19
p_hat_best_models$kNN = p_hat_L$kNN[,i]
p_hat_best_models$CART = p_hat_L$CART
i = 8
p_hat_best_models$RF =  p_hat_L$RF[,i]
i = 6
p_hat_best_models$Boost = p_hat_L$Boost[,i]
p_hat_best_models = as.data.frame(p_hat_best_models)
nmethod = length(p_hat_best_models) 
auc_array = NULL
plot(c(0,1),c(0,1),xlab='FPR',ylab='TPR',main="ROC curve",cex.lab=1,type="n")
for(i in 1:ncol(p_hat_best_models)) {
  pred = prediction(p_hat_best_models[,i], y)
  perf = performance(pred, measure = "tpr", x.measure = "fpr") 
  auc = performance(pred, measure = "auc")
  auc_array = c(auc_array, auc@y.values)
  lines(perf@x.values[[1]], perf@y.values[[1]],col=i)
}
abline(0,1,lty=2) 
legend("topleft",legend=names(p_hat_best_models),col=1:nmethod,lty=rep(1,nmethod), cex = 0.5)
auc_array = as.data.frame(auc_array)
names(auc_array) = names(p_hat_best_models)
print(auc_array)

#### Aggreggate Methods?#####
##############################
# Probably out of scope; I'd rather not implement this
# We could use the best of each type of model
# We could average all probabilities, and get outcome by applying threshold
# How would we get the probability using majority vote? Averaging probabilities for those in favor of the outcome?
# Store probabilities
# Store predictions (using threshold on probabilities)
# Calculate deviance for aggreggate method?
# We would have to use ALL the best models to predict tournament results

####     Final Model     #####
##############################
# See which model is best (minimum deviance)
which.min(all_losses)
# Result: Boosting #6 (Logistic Regression is very close & has a better accuracy with validation set)
idv = c(2,4) #Depth
ntv = c(1000,5000) #Number of trees
shv = c(0.1,0.01) #Shrinking
setboost = expand.grid(idv,ntv,shv)
colnames(setboost) = c("tdepth","ntree","shrink")
# Retrain best model with train + validation
# Boosting
i = 6
fboost = gbm(Outcome~., data=train_validation_set_numerical, distribution="bernoulli",
             n.trees=setboost[i,2], 
             interaction.depth=setboost[i,1], 
             shrinkage=setboost[i,3])
p_hat = predict(fboost, newdata=test_set_numerical,n.trees=setboost[i,2], type="response")
p_hat_boost = p_hat
y_hat_boost = p_hat
y_hat_boost[p_hat >= threshold] = 1
y_hat_boost[p_hat < threshold] = 0
# Logistic Regression
lr_fit = glm(Outcome~., train_validation_set, family=binomial(link = "logit"))
p_hat_lr = predict(lr_fit, test_set, type="response")
y_hat_lr = matrix(p_hat_lr,ncol = 1)
y_hat_lr[p_hat_lr >= threshold] = 1
y_hat_lr[p_hat_lr < threshold] = 0

# Use test set to give a final accuracy description of the model
y = as.numeric(test_set$Outcome)-1
cm = cm = table(predictions = y_hat_boost, actual = y)
accuracy_Boost = accuracy_from_cm(cm)
# specificity: true positive rate (TPR) = TP / (TP + FN)
specificity = cm[4] / (cm[4] + cm[3])
# sensitivity: true negative rate (TNR) = TN / (TN + FP)
sensitivity = cm[1] / (cm[1] + cm[2])
print("Boosting:")
print(cm)
print(accuracy_Boost)
print(specificity)
print(sensitivity)
cm = table(predictions = y_hat_lr, actual = y)
accuracy_lr = accuracy_from_cm(cm)
# specificity: true positive rate (TPR) = TP / (TP + FN)
specificity = cm[4] / (cm[4] + cm[3])
# sensitivity: true negative rate (TNR) = TN / (TN + FP)
sensitivity = cm[1] / (cm[1] + cm[2])
print("Logistic Regression:")
print(cm)
print(accuracy_lr)
print(specificity)
print(sensitivity)

#### Torunament Prediction #####
################################
# Retrain best model with entire data set (train + validation + test)
idv = c(2,4) #Depth
ntv = c(1000,5000) #Number of trees
shv = c(0.1,0.01) #Shrinking
setboost = expand.grid(idv,ntv,shv)
colnames(setboost) = c("tdepth","ntree","shrink")
# Retrain best model with train + validation + test (tennis_data)
load.Rdata(filename="Results.Rdata", objname = "tennis_data") 
names(tennis_data)[6] = "BestOf"
tennis_data$Tournament = as.factor(tennis_data$Tournament)
tennis_data$Year = as.factor(tennis_data$Year)
tennis_data$Court = as.factor(tennis_data$Court)
tennis_data$Surface = as.factor(tennis_data$Surface)
tennis_data$Round = as.factor(tennis_data$Round)
tennis_data$BestOf = as.factor(tennis_data$BestOf)
tennis_data$Player1 = as.factor(tennis_data$Player1)
tennis_data$Player2 = as.factor(tennis_data$Player2)
tennis_data$Outcome = as.factor(tennis_data$Outcome)
tennis_data$P1Rank = as.numeric(tennis_data$P1Rank)
tennis_data$P2Rank = as.numeric(tennis_data$P2Rank)
tennis_data$P1Pts = as.numeric(tennis_data$P1Pts)
tennis_data$P2Pts = as.numeric(tennis_data$P2Pts)
tennis_data_set = tennis_data
tennis_data_set = as.data.frame(tennis_data_set %>%
                                       select(Outcome,P1Pts,P2Pts, 
                                              Player1Srv1Wp,Player1GamesWp,Player1MatchesWp,Player1SetWp,
                                              Player2Srv1Wp,Player2GamesWp,Player2MatchesWp,Player2SetWp))
tennis_data_set_numerical = tennis_data_set
tennis_data_set_numerical$Outcome = as.numeric(tennis_data_set$Outcome)-1
tennis_data_set_numeric = tennis_data_set
tennis_data_set_numeric$Player1Srv1Wp = as.numeric(tennis_data_set_numeric$Player1Srv1Wp)
tennis_data_set_numeric$Player1GamesWp = as.numeric(tennis_data_set_numeric$Player1GamesWp)
tennis_data_set_numeric$Player1MatchesWp = as.numeric(tennis_data_set_numeric$Player1MatchesWp)
tennis_data_set_numeric$Player1SetWp = as.numeric(tennis_data_set_numeric$Player1SetWp)
tennis_data_set_numeric$Player2Srv1Wp = as.numeric(tennis_data_set_numeric$Player2Srv1Wp)
tennis_data_set_numeric$Player2GamesWp = as.numeric(tennis_data_set_numeric$Player2GamesWp)
tennis_data_set_numeric$Player2MatchesWp = as.numeric(tennis_data_set_numeric$Player2MatchesWp)
tennis_data_set_numeric$Player2SetWp = as.numeric(tennis_data_set_numeric$Player2SetWp)
i = 6
model = "boost"
if (model == "boost") {
  final_model = gbm(Outcome~., data=tennis_data_set_numerical, distribution="bernoulli",
               n.trees=setboost[i,2], 
               interaction.depth=setboost[i,1], 
               shrinkage=setboost[i,3])
} else {
  final_model = glm(Outcome~., tennis_data_set_numeric, family=binomial(link = "logit"))
}

# Create variables needed for tournament simulation
french_open_data = read.csv("FrenchOpen2020_1stRound.csv")
french_open_data = as.data.frame(french_open_data)
french_open_data$Player1 = as.character(french_open_data$Player1)
french_open_data$Player2 = as.character(french_open_data$Player2)
p1_column_names = c("Player1","P1Rank","P1Pts","Player1ACp","Player1DFp","Player1Srv1p","Player1Srv1Wp",
                    "Player1Srv2Wp","Player1Srv1ReWp","Player1Srv2ReWp","Player1GamesWp","Player1MatchesWp","Player1SetWp")
p2_column_names = c("Player2","P2Rank","P2Pts","Player2ACp","Player2DFp","Player2Srv1p","Player2Srv1Wp",
                    "Player2Srv2Wp","Player2Srv1ReWp","Player2Srv2ReWp","Player2GamesWp","Player2MatchesWp","Player2SetWp")

#### Torunament Simulation #####
################################
# Extracting French Open players list
players_list = c(french_open_data$Player1,french_open_data$Player2)
players_rank = c(french_open_data$P1Rank,french_open_data$P2Rank)
players_type = c(rep(1,nrow(french_open_data)),rep(2,nrow(french_open_data)))
players_row = c(1:64,1:64)
players = list()
players$name = players_list
players$rank = players_rank
players$column = players_type
players$row = players_row
players = as.data.frame(players)
players = players[order(players$rank),]
players$position = (players$row)*2-1 + (players$column == 2)*1
# Extracting possible tournament placements
positions_taken = players$position[1:32]
positions_available = 1:128
positions_available = positions_available[-positions_taken]
# Running simulations
simulations = 100
all_simulation_results = as.data.frame(players$name)
names(all_simulation_results) = "name"
all_simulation_results$rank = players$rank
for (count in 1:simulations) {
  # Generate simulation
  french_open_simulation = create_tournament_simulation(french_open_data,players,positions_available,p1_column_names,p2_column_names)
  # Execute simulation
  simulation_results = simulate_tournament(french_open_simulation,final_model,setboost,p1_column_names,p2_column_names,players,model)
  all_simulation_results = cbind(all_simulation_results,simulation_results$results)
  names(all_simulation_results)[count + 2] = sprintf("sim_%d",count)
}
simmulation_summary = all_simulation_results[,1:2]
simmulation_summary$winner = 0
simmulation_summary$final = 0
simmulation_summary$semi_final = 0
simmulation_summary$quarter_final = 0
simmulation_summary$round_4 = 0
simmulation_summary$round_3 = 0
simmulation_summary$round_2 = 0
simmulation_summary$round_1 = 0
for (i in 1:nrow(simmulation_summary)) {
  simmulation_summary$winner[i] = sum((all_simulation_results[i,3:(simulations+2)] == "winner")*1) / simulations * 100
  simmulation_summary$final[i] = sum((all_simulation_results[i,3:(simulations+2)] == "final")*1) / simulations * 100
  simmulation_summary$semi_final[i] = sum((all_simulation_results[i,3:(simulations+2)] == "semi_finals")*1) / simulations * 100
  simmulation_summary$quarter_final[i] = sum((all_simulation_results[i,3:(simulations+2)] == "quarter_finals")*1) / simulations * 100
  simmulation_summary$round_4[i] = sum((all_simulation_results[i,3:(simulations+2)] == "4th_round")*1) / simulations * 100
  simmulation_summary$round_3[i] = sum((all_simulation_results[i,3:(simulations+2)] == "3rd_round")*1) / simulations * 100
  simmulation_summary$round_2[i] = sum((all_simulation_results[i,3:(simulations+2)] == "2nd_round")*1) / simulations * 100
  simmulation_summary$round_1[i] = sum((all_simulation_results[i,3:(simulations+2)] == "1st_round")*1) / simulations * 100
}
simmulation_summary_boost = simmulation_summary

#### Simulation Functions  #####
################################
#### Generating Simulation
################################
create_tournament_simulation = function(french_open_data,players,positions_available,p1_column_names,p2_column_names) {
  # Creating new French Open simulation
  french_open_data_simulation = french_open_data
  # randomizing placement, except for top 32 players
  placement = sample(positions_available)
  players$placement = players$position
  players$placement[players$rank > 32] = placement
  # Copying values into French Open simulation
  for (i in 1:nrow(players)) {
    simulation_set_col = if (players$placement[i] %% 2 == 0) 2 else 1
    simulation_set_row = if (simulation_set_col == 2) players$placement[i] / 2 else (players$placement[i] + 1) / 2
    french_open_set_col = if (players$position[i] %% 2 == 0) 2 else 1
    french_open_set_row = if (french_open_set_col == 2) players$position[i] / 2 else (players$position[i] + 1) / 2
    if (simulation_set_col == 1) {
      if (french_open_set_col == 1) {
        #Copy player 1 into player 1
        for (j in 1:length(p1_column_names)) {
          french_open_data_simulation[simulation_set_row,p1_column_names[j]] = 
            french_open_data[french_open_set_row,p1_column_names[j]]
        }
      }
      else {
        #Copyt player 2 into player 1
        for (j in 1:length(p1_column_names)) {
          french_open_data_simulation[simulation_set_row,p1_column_names[j]] = 
            french_open_data[french_open_set_row,p2_column_names[j]]
        }
      }
    }
    else {
      if (french_open_set_col == 1) {
        #Copy player 1 into player 2
        for (j in 1:length(p2_column_names)) {
          french_open_data_simulation[simulation_set_row,p2_column_names[j]] = 
            french_open_data[french_open_set_row,p1_column_names[j]]
        }
      }
      else {
        #Copyt player 2 into player 2
        for (j in 1:length(p2_column_names)) {
          french_open_data_simulation[simulation_set_row,p2_column_names[j]] = 
            french_open_data[french_open_set_row,p2_column_names[j]]
        }
      }
    }
  }
  return (french_open_data_simulation)
}
#### Executing Simulation
################################
simulate_tournament = function(french_open_data,final_model,setboost,p1_column_names,p2_column_names,players,model_selected) {
  simulation_results = as.data.frame(players[,1])
  names(simulation_results) = "name"
  current_round = "1st_round"
  simulation_results$results = current_round
  # 1st Round - 64 games
  french_open_data_1st_round = french_open_data
  tournament_1st_round = as.data.frame(french_open_data_1st_round %>%
                                         select(P1Pts,P2Pts, 
                                                Player1Srv1Wp,Player1GamesWp,Player1MatchesWp,Player1SetWp,
                                                Player2Srv1Wp,Player2GamesWp,Player2MatchesWp,Player2SetWp))
  i = 6
  if (model_selected == "boost") {
    prob_prediction = predict(final_model, newdata=tournament_1st_round,n.trees=setboost[i,2], type="response")
  } else {
    prob_prediction = predict(final_model, tournament_1st_round, type="response")
  }
  outcome_prediction_1st_round = prob_prediction
  outcome_prediction_1st_round[prob_prediction >= threshold] = 1
  outcome_prediction_1st_round[prob_prediction < threshold] = 0
  
  # 2nd Round - 32 games
  current_round = "2nd_round"
  french_open_data_2nd_round = french_open_data
  french_open_data_2nd_round$Round = rep("2nd Round",nrow(french_open_data_2nd_round))
  for (i in 1:32) {
    n = i * 2 - 1
    if (outcome_prediction_1st_round[n] == 1) {
      # Copy player 1 into player 1 position
      for (j in 1:length(p1_column_names)) {
        french_open_data_2nd_round[i,p1_column_names[j]] = french_open_data_1st_round[n,p1_column_names[j]]
      }
    }
    else {
      # Copy player 2 into player 1 position
      for (j in 1:length(p2_column_names)) {
        french_open_data_2nd_round[i,p1_column_names[j]] = french_open_data_1st_round[n,p2_column_names[j]]
      }
    }
    n = n + 1
    if (outcome_prediction_1st_round[n] == 1) {
      # Copy player 1 into player 2 position
      for (j in 1:length(p1_column_names)) {
        french_open_data_2nd_round[i,p2_column_names[j]] = french_open_data_1st_round[n,p1_column_names[j]]
      }
    }
    else {
      # Copy player 2 into player 2 position
      for (j in 1:length(p2_column_names)) {
        french_open_data_2nd_round[i,p2_column_names[j]] = french_open_data_1st_round[n,p2_column_names[j]]
      }
    }
    simulation_results$results[match(french_open_data_2nd_round$Player1[i],simulation_results$name)] = current_round
    simulation_results$results[match(french_open_data_2nd_round$Player2[i],simulation_results$name)] = current_round
  }
  french_open_data_2nd_round = french_open_data_2nd_round[1:32,]
  tournament_2nd_round = as.data.frame(french_open_data_2nd_round %>%
                                         select(P1Pts,P2Pts, 
                                                Player1Srv1Wp,Player1GamesWp,Player1MatchesWp,Player1SetWp,
                                                Player2Srv1Wp,Player2GamesWp,Player2MatchesWp,Player2SetWp))
  i = 6
  if (model_selected == "boost") {
    prob_prediction = predict(final_model, newdata=tournament_2nd_round,n.trees=setboost[i,2], type="response")
  } else {
    prob_prediction = predict(final_model, tournament_2nd_round, type="response")
  }
  outcome_prediction_2nd_round = prob_prediction
  outcome_prediction_2nd_round[prob_prediction >= threshold] = 1
  outcome_prediction_2nd_round[prob_prediction < threshold] = 0
  
  # 3rd Round - 16 games
  current_round = "3rd_round"
  french_open_data_3rd_round = french_open_data
  french_open_data_3rd_round$Round = rep("3rd Round",nrow(french_open_data_3rd_round))
  for (i in 1:16) {
    n = i * 2 - 1
    if (outcome_prediction_2nd_round[n] == 1) {
      # Copy player 1 into player 1 position
      for (j in 1:length(p1_column_names)) {
        french_open_data_3rd_round[i,p1_column_names[j]] = french_open_data_2nd_round[n,p1_column_names[j]]
      }
    }
    else {
      # Copy player 2 into player 1 position
      for (j in 1:length(p2_column_names)) {
        french_open_data_3rd_round[i,p1_column_names[j]] = french_open_data_2nd_round[n,p2_column_names[j]]
      }
    }
    n = n + 1
    if (outcome_prediction_2nd_round[n] == 1) {
      # Copy player 1 into player 2 position
      for (j in 1:length(p1_column_names)) {
        french_open_data_3rd_round[i,p2_column_names[j]] = french_open_data_2nd_round[n,p1_column_names[j]]
      }
    }
    else {
      # Copy player 2 into player 2 position
      for (j in 1:length(p2_column_names)) {
        french_open_data_3rd_round[i,p2_column_names[j]] = french_open_data_2nd_round[n,p2_column_names[j]]
      }
    }
    simulation_results$results[match(french_open_data_2nd_round$Player1[i],simulation_results$name)] = current_round
    simulation_results$results[match(french_open_data_2nd_round$Player2[i],simulation_results$name)] = current_round
  }
  french_open_data_3rd_round = french_open_data_3rd_round[1:16,]
  tournament_3rd_round = as.data.frame(french_open_data_3rd_round %>%
                                         select(P1Pts,P2Pts, 
                                                Player1Srv1Wp,Player1GamesWp,Player1MatchesWp,Player1SetWp,
                                                Player2Srv1Wp,Player2GamesWp,Player2MatchesWp,Player2SetWp))
  i = 6
  if (model_selected == "boost") {
    prob_prediction = predict(final_model, newdata=tournament_3rd_round,n.trees=setboost[i,2], type="response")
  } else {
    prob_prediction = predict(final_model, tournament_3rd_round, type="response")
  }
  outcome_prediction_3rd_round = prob_prediction
  outcome_prediction_3rd_round[prob_prediction >= threshold] = 1
  outcome_prediction_3rd_round[prob_prediction < threshold] = 0
  
  # 4th Round - 8 games
  current_round = "4th_round"
  french_open_data_4th_round = french_open_data
  french_open_data_4th_round$Round = rep("4th Round",nrow(french_open_data_4th_round))
  for (i in 1:8) {
    n = i * 2 - 1
    if (outcome_prediction_3rd_round[n] == 1) {
      # Copy player 1 into player 1 position
      for (j in 1:length(p1_column_names)) {
        french_open_data_4th_round[i,p1_column_names[j]] = french_open_data_3rd_round[n,p1_column_names[j]]
      }
    }
    else {
      # Copy player 2 into player 1 position
      for (j in 1:length(p2_column_names)) {
        french_open_data_4th_round[i,p1_column_names[j]] = french_open_data_3rd_round[n,p2_column_names[j]]
      }
    }
    n = n + 1
    if (outcome_prediction_3rd_round[n] == 1) {
      # Copy player 1 into player 2 position
      for (j in 1:length(p1_column_names)) {
        french_open_data_4th_round[i,p2_column_names[j]] = french_open_data_3rd_round[n,p1_column_names[j]]
      }
    }
    else {
      # Copy player 2 into player 2 position
      for (j in 1:length(p2_column_names)) {
        french_open_data_4th_round[i,p2_column_names[j]] = french_open_data_3rd_round[n,p2_column_names[j]]
      }
    }
    simulation_results$results[match(french_open_data_2nd_round$Player1[i],simulation_results$name)] = current_round
    simulation_results$results[match(french_open_data_2nd_round$Player2[i],simulation_results$name)] = current_round
  }
  french_open_data_4th_round = french_open_data_4th_round[1:8,]
  tournament_4th_round = as.data.frame(french_open_data_4th_round %>%
                                         select(P1Pts,P2Pts, 
                                                Player1Srv1Wp,Player1GamesWp,Player1MatchesWp,Player1SetWp,
                                                Player2Srv1Wp,Player2GamesWp,Player2MatchesWp,Player2SetWp))
  i = 6
  if (model_selected == "boost") {
    prob_prediction = predict(final_model, newdata=tournament_4th_round,n.trees=setboost[i,2], type="response")
  } else {
    prob_prediction = predict(final_model, tournament_4th_round, type="response")
  }
  outcome_prediction_4th_round = prob_prediction
  outcome_prediction_4th_round[prob_prediction >= threshold] = 1
  outcome_prediction_4th_round[prob_prediction < threshold] = 0
  
  # Quarterfinals - 4 games
  current_round = "quarter_finals"
  french_open_data_QF_round = french_open_data
  french_open_data_QF_round$Round = rep("Quarter Finals",nrow(french_open_data_QF_round))
  for (i in 1:4) {
    n = i * 2 - 1
    if (outcome_prediction_4th_round[n] == 1) {
      # Copy player 1 into player 1 position
      for (j in 1:length(p1_column_names)) {
        french_open_data_QF_round[i,p1_column_names[j]] = french_open_data_4th_round[n,p1_column_names[j]]
      }
    }
    else {
      # Copy player 2 into player 1 position
      for (j in 1:length(p2_column_names)) {
        french_open_data_QF_round[i,p1_column_names[j]] = french_open_data_4th_round[n,p2_column_names[j]]
      }
    }
    n = n + 1
    if (outcome_prediction_4th_round[n] == 1) {
      # Copy player 1 into player 2 position
      for (j in 1:length(p1_column_names)) {
        french_open_data_QF_round[i,p2_column_names[j]] = french_open_data_4th_round[n,p1_column_names[j]]
      }
    }
    else {
      # Copy player 2 into player 2 position
      for (j in 1:length(p2_column_names)) {
        french_open_data_QF_round[i,p2_column_names[j]] = french_open_data_4th_round[n,p2_column_names[j]]
      }
    }
    simulation_results$results[match(french_open_data_2nd_round$Player1[i],simulation_results$name)] = current_round
    simulation_results$results[match(french_open_data_2nd_round$Player2[i],simulation_results$name)] = current_round
  }
  french_open_data_QF_round = french_open_data_QF_round[1:4,]
  tournament_QF_round = as.data.frame(french_open_data_QF_round %>%
                                        select(P1Pts,P2Pts, 
                                               Player1Srv1Wp,Player1GamesWp,Player1MatchesWp,Player1SetWp,
                                               Player2Srv1Wp,Player2GamesWp,Player2MatchesWp,Player2SetWp))
  i = 6
  if (model_selected == "boost") {
    prob_prediction = predict(final_model, newdata=tournament_QF_round,n.trees=setboost[i,2], type="response")
  } else {
    prob_prediction = predict(final_model, tournament_QF_round, type="response")
  }
  outcome_prediction_QF_round = prob_prediction
  outcome_prediction_QF_round[prob_prediction >= threshold] = 1
  outcome_prediction_QF_round[prob_prediction < threshold] = 0
  
  # Semifinals - 2 games
  current_round = "semi_finals"
  french_open_data_SF_round = french_open_data
  french_open_data_SF_round$Round = rep("Semi Finals",nrow(french_open_data_SF_round))
  for (i in 1:2) {
    n = i * 2 - 1
    if (outcome_prediction_QF_round[n] == 1) {
      # Copy player 1 into player 1 position
      for (j in 1:length(p1_column_names)) {
        french_open_data_SF_round[i,p1_column_names[j]] = french_open_data_QF_round[n,p1_column_names[j]]
      }
    }
    else {
      # Copy player 2 into player 1 position
      for (j in 1:length(p2_column_names)) {
        french_open_data_SF_round[i,p1_column_names[j]] = french_open_data_QF_round[n,p2_column_names[j]]
      }
    }
    n = n + 1
    if (outcome_prediction_QF_round[n] == 1) {
      # Copy player 1 into player 2 position
      for (j in 1:length(p1_column_names)) {
        french_open_data_SF_round[i,p2_column_names[j]] = french_open_data_QF_round[n,p1_column_names[j]]
      }
    }
    else {
      # Copy player 2 into player 2 position
      for (j in 1:1) {
        french_open_data_SF_round[i,p2_column_names[j]] = french_open_data_QF_round[n,p2_column_names[j]]
      }
    }
    simulation_results$results[match(french_open_data_2nd_round$Player1[i],simulation_results$name)] = current_round
    simulation_results$results[match(french_open_data_2nd_round$Player2[i],simulation_results$name)] = current_round
  }
  french_open_data_SF_round = french_open_data_SF_round[1:2,]
  tournament_SF_round = as.data.frame(french_open_data_SF_round %>%
                                        select(P1Pts,P2Pts, 
                                               Player1Srv1Wp,Player1GamesWp,Player1MatchesWp,Player1SetWp,
                                               Player2Srv1Wp,Player2GamesWp,Player2MatchesWp,Player2SetWp))
  i = 6
  if (model_selected == "boost") {
    prob_prediction = predict(final_model, newdata=tournament_SF_round,n.trees=setboost[i,2], type="response")
  } else {
    prob_prediction = predict(final_model, tournament_SF_round, type="response")
  }
  outcome_prediction_SF_round = prob_prediction
  outcome_prediction_SF_round[prob_prediction >= threshold] = 1
  outcome_prediction_SF_round[prob_prediction < threshold] = 0
  
  # Final - 1 game
  current_round = "final"
  french_open_data_Final_round = french_open_data
  french_open_data_Final_round$Round = rep("Final",nrow(french_open_data_Final_round))
  for (i in 1:1) {
    n = i * 2 - 1
    if (outcome_prediction_SF_round[n] == 1) {
      # Copy player 1 into player 1 position
      for (j in 1:length(p1_column_names)) {
        french_open_data_Final_round[i,p1_column_names[j]] = french_open_data_SF_round[n,p1_column_names[j]]
      }
    }
    else {
      # Copy player 2 into player 1 position
      for (j in 1:length(p2_column_names)) {
        french_open_data_Final_round[i,p1_column_names[j]] = french_open_data_SF_round[n,p2_column_names[j]]
      }
    }
    n = n + 1
    if (outcome_prediction_SF_round[n] == 1) {
      # Copy player 1 into player 2 position
      for (j in 1:length(p1_column_names)) {
        french_open_data_Final_round[i,p2_column_names[j]] = french_open_data_SF_round[n,p1_column_names[j]]
      }
    }
    else {
      # Copy player 2 into player 2 position
      for (j in 1:length(p2_column_names)) {
        french_open_data_Final_round[i,p2_column_names[j]] = french_open_data_SF_round[n,p2_column_names[j]]
      }
    }
    simulation_results$results[match(french_open_data_2nd_round$Player1[i],simulation_results$name)] = current_round
    simulation_results$results[match(french_open_data_2nd_round$Player2[i],simulation_results$name)] = current_round
  }
  french_open_data_Final_round = french_open_data_Final_round[1,]
  tournament_Final_round = as.data.frame(french_open_data_Final_round %>%
                                           select(P1Pts,P2Pts, 
                                                  Player1Srv1Wp,Player1GamesWp,Player1MatchesWp,Player1SetWp,
                                                  Player2Srv1Wp,Player2GamesWp,Player2MatchesWp,Player2SetWp))
  i = 6
  if (model_selected == "boost") {
    prob_prediction = predict(final_model, newdata=tournament_Final_round,n.trees=setboost[i,2], type="response")
  } else {
    prob_prediction = predict(final_model, tournament_Final_round, type="response")
  }
  outcome_prediction_Final_round = prob_prediction
  outcome_prediction_Final_round[prob_prediction >= threshold] = 1
  outcome_prediction_Final_round[prob_prediction < threshold] = 0
  
  if (outcome_prediction_Final_round == 1) {
    winner = french_open_data_Final_round$Player1
  } else {
    winner = french_open_data_Final_round$Player2
  }
  simulation_results$results[match(winner,simulation_results$name)] = "winner"
  return (simulation_results)
}




