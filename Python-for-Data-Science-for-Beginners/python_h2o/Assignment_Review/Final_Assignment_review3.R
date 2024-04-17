##### Final project

# Step 1: Load project data and preprocessing/feature engneering -------------------------------------------------------

library(h2o)
h2o.init()
library(dplyr)

url <- "http://coursera.h2o.ai/house_data.3487.csv"
data <- read.csv(url, stringsAsFactors = F)    # it takes a while to read raw data

# Create new variable "home_age", number of years since is was built: 2019 - yr_built, 
# in order to measure the physical condition of house, but if renovated then replace yr_built with this.

data <- data %>%
  mutate(# split year, month
    year = substr(date, 1, 4),
    month = substr(date, 5, 6),
    
    # create aging of house
    new_built = ifelse(yr_renovated != 0, yr_renovated, yr_built),
    home_age = 2019 - new_built,
    
    # convert zipcode, waterfront, view to factor, year, month
    zipcode = as.factor(zipcode),
    waterfront = as.factor(waterfront),
    view = as.factor(view),
    year = as.factor(year),
    month = as.factor(month)) %>%
  # drop unnecessary variables
  select(-c(id, date, yr_renovated, yr_built, lat, long, new_built))

data <- as.h2o(data)
parts <- h2o.splitFrame(data, 0.9, seed = 123)
train <- parts[[1]]   
test <- parts[[2]]     
dim(train)  # 19462 / 18
dim(test)   # 2151 / 18

y <- 'price'
x <- setdiff(colnames(train), y)

# Step 2: Train models -----------------------------------------------------------

folds <- 5     # number of folds for all models
N <- 12345     # seed numbers for all models

#
# model 1: GBM
#
# This is a GBM model with random hyper-parameter search to tune learn_rate, max_depth, ntrees, and min_rows and
# the values are 0.07, 6, 200, and 5 respectively. Model was evaluated based on RMSE criteria in cross-validation 
# data set.  The mean value of the RMSE for the cross-validation data set is  $122,586 which is lower than
# target $123,000.
#

my_gbm <- h2o.gbm( x = x, y = y, training_frame = train,
                   nfolds = folds, 
                   seed = N, 
                   fold_assignment = 'Modulo',
                   keep_cross_validation_predictions = TRUE,
                   # trained hyper-parameters
                   ntrees = 200, max_depth = 6, min_rows = 5, learn_rate = 0.07) 
# rmse on cv data
h2o.rmse(my_gbm, xval = TRUE)  #122,586
# save models
h2o.saveModel(my_gbm, path='./models', force = TRUE)

#
# model 2: RandomForest
#
# This is a RandomForest model with random hyper-parameter search to tune ntrees, mtries, max_depth, min_rows, and
# sample rate. And the optimized values are 150, 6, 12, 2, and 1 respectively. Model was evaluated based on RMSE criteria in cross-validation 
# data set.  The mean value of the RMSE for the cross-validation data set is  $131,596 which is a bit greater than
# target $123,000.  

my_rf <- h2o.randomForest(x = x, y = y, training_frame = train, nfolds = folds, seed = N,
                          stopping_metric = 'RMSE',
                          stopping_tolerance = 1e-5,
                          max_runtime_secs = 3600,
                          fold_assignment = 'Modulo',
                          keep_cross_validation_predictions = TRUE,
                          # trained hyper-parameters,
                          ntrees = 150, mtries = 6, max_depth = 12, min_rows = 2, sample_rate = 1)
# rmse on cv data
h2o.rmse(my_rf, xval = TRUE) # 131,596
# save models
h2o.saveModel(my_rf, path='./models', force = TRUE)

#
# model 3: GLM
#
# This is a GLM model with random hyper-parameter search to tune L1, L2 regularization parameters alpha and
# lambda. And the optimized values are 0.0039 and 2.0e-4. Model was evaluated based on RMSE criteria in cross-validation 
# data set.  The mean value of the RMSE for the cross-validation data set is  $162,076 which is welll above than
# target $123,000.  
#

my_glm <- h2o.glm(x = x, y = y, training_frame = train, nfolds = folds, seed = N,
                  standardize = TRUE,
                  #family = "gaussian", 
                  fold_assignment = 'Modulo',
                  keep_cross_validation_predictions = TRUE,
                  # trained hyper-params
                  alpha = 0.0039, lambda = 2.0e-4)
# rmse on cv data
h2o.rmse(my_glm, xval=TRUE) #162,076
# save models
h2o.saveModel(my_glm, path='./models', force = TRUE)

#
# Model 4: DeepLearning
# This is a DL model with random hyper-parameter search to tune activation, l1, l2, input_dropout_ration,
# hidden, and hidden_dropout_ratios. And the optimized values are REctifierWithDropout, 0, 0, 0.1, (400, 400),
# and (0.6, 0.6) respectively. Model was evaluated based on RMSE criteria in cross-validation 
# data set.  The mean value of the RMSE for the cross-validation data set is  $132,736 which is greater than
# target $123,000.  

my_dl <- h2o.deeplearning(x = x, y = y, training_frame = train, nfolds = folds, seed = N,
                          standardize = TRUE,
                          stopping_metric = "RMSE",
                          stopping_tolerance = 1e-5,
                          stopping_rounds = 2,
                          fold_assignment = 'Modulo',
                          keep_cross_validation_predictions = TRUE,
                          # trained hyper-parameters
                          activation = 'RectifierWithDropout', l1 = 0.0, l2 = 0.0, 
                          input_dropout_ratio = 0.1, hidden = c(400, 400),
                          hidden_dropout_ratios = c(0.6, 0.6))
# rmse
h2o.rmse(my_dl, xval=TRUE) #132,736
# save models
h2o.saveModel(my_dl, path='./models', force = TRUE)

#
# Step 3: Train Stacked Ensemble model
#

model_ids <- list(my_gbm@model_id, my_rf@model_id, my_glm@model_id, my_dl@model_id)
my_SE <- h2o.stackedEnsemble(x = x, y = y, training_frame = train, seed = N,
                             model_id = "SE_4models",
                             metalearner_algorithm = 'deeplearning',
                             base_models = model_ids)
# save models
h2o.saveModel(my_SE, path='./models', force = TRUE)

#
# Step 4: Select best performance model
# RMSE for stacked ensemble model is not available on cross validation dta set, thus test data set is used
# for model performance comparison
#

models <- c(my_gbm, my_rf, my_glm, my_dl, my_SE)

# rmse on cv data
sapply(models, h2o.rmse, xval=TRUE)
# GBM:122586, RandomForest:131596, GLM:162076, DeepLearning:132736, StackedEnsemble:NULL

# rmse on test data
perf <- lapply(models, h2o.performance, test)
sapply(perf, h2o.rmse)  
# GBM:129757, RandomForest:132273, GLM:152479, DeepLearning:120243, StackedEnsemble:122224


#
# Final step: Summarize the results 
# Both Deeplearning and Ensemble model showing best results against target value but Deeplearning model is selected
# because it performed better than Ensemble model.  Additionally, GBM and RandomForest showing a bit of overfitting 
# (rmse on test data are greater than train data), while GLM and Deeplearning look better in unseen data.