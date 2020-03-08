##############################
#####  PACKAGE SETUP     #####
##############################
#List the packages we need, install if missing, then load all of them
PackageList =c('tidyverse')
NewPackages=PackageList[!(PackageList %in%
                            installed.packages()[,"Package"])]
if(length(NewPackages)) install.packages(NewPackages)
lapply(PackageList,require,character.only=TRUE) #array function

# Set Seed
set.seed(6992) 

#### SETUP ######
################
# Data is located in Results.Rdata

# Cleanse Data (Rename Columns/Drop Columns e.t.c.)
# Change variables to factors / numerical values

# Partition Data to Train/Test/Validation
# Use 60% for Train (to be used for models)
# Use 20% for Validation (to evaluate models with Deviance loss)
# Use 20% for Test (once model is selected, report Accuracy with Test set)

# Use threshold = 0.5 for all models

#### Logistic Regression #####
##############################
# Run LASSO to evaluate variable selection
# Evaluate LASSO with min lambda and lambda 1 standard error
# Run GLM & Inspect

# Store probabilities

# Store predictions (using threshold on probabilities) 

#### K Nearest Neighbor #####
#############################
# Normalize data (or better yet standardized with mean = 0 and sd = 1)

# Run KNN using CV or maybe Caret internal CV to find optimal K

# Store probabilities

# Store predictions (using threshold on probabilities)

#### Classification Tree #####
##############################
# Build Complex Tree
# control=rpart.control(minsplit = 5,cp=0.00025,xval=10))

# Prune Complex Tree based on best cp value considering standard error

# Store probabilities

# Store predictions (using threshold on probabilities)

#### Random Forest #####
########################
# Build RF with different paramter values
# p = dim(data_set)[2]-1
# mtryv = c(p,sqrt(p)) #number of variables
# ntreev = c(50,100,250,500) #number of trees
# setrf = expand.grid(mtryv,ntreev) 
# colnames(setrf)=c("mtry","ntree") 

# Store probabilities - all models

# Store predictions (using threshold on probabilities) - all models

####    Boosting   #####
########################
# Build Boosting tree with different paramter values
# idv = c(2,4) #Depth
# ntv = c(1000,5000) #Number of trees
# shv = c(0.1,0.01) #Shrinking
# setboost = expand.grid(idv,ntv,shv)
# colnames(setboost) = c("tdepth","ntree","shrink")

# Store probabilities - all models

# Store predictions (using threshold on probabilities) - all models


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

# See which model is best (minimum deviance)

# Retrain best model with train + validation

# Use test set to give a final accuracy description of the model


#### Torunament Prediction #####
################################
# Retrain best model with entire data set (train + validation + test)

# Use tournament data (to be provided by Carly / Sylvia)
# Predict brackets and create following rounds iteratively

# Summarize French Open results with each brakcet winners up to final winner
