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
# Read Data

# Cleanse Data (Rename Columns/Drop Columns e.t.c.)

# Partition Data to Train/Test/Validation


#### Logistic Regression #####
##############################
# Run GLM & Inspect

# Predict against different possible formulas

# Get Deviance, AIC, BIC

# Pick Optimal Formula

# Store Predictions

# Store Probability & maybe Confusion Matrix?


#### K Nearest Neighbor #####
#############################
# Run KNN using CV or maybe Caret internal CV to find optimal K

# Store Predictions

# Store Probability & maybe Confusion Matrix?


#### Classification Tree #####
##############################
# Build Complex Tree

# Prune Complex Tree

# Rebuild Tree based on Better Pruning

# Store Predictions

# Store Probability & maybe Confusion Matrix?


#### Random Forest #####
########################
# Build RF with different paramter values

# Find Optimal Values

# Store Predictions

# Store Probability & maybe Confusion Matrix?


#### Aggreggate Methods? #####
##############################
# Potentially use Average & Majority Votes? Other aggreggate methods?

# Store Predictions

# Store Probability & maybe Confusion Matrix?


#### Torunament Prediction #####
################################
# Use Optimal Method to Predict on Tournament (Bracket by Bracket? / All Possible Combos?)

# TODO: How da F does the rules of French Open matching work again?

# Carlos test commit
