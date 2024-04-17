rm(list=ls())
library(h2o)
#install.packages("dplyr")
library(dplyr)
# STEP 1
h2o.init()
# H2O cluster total memory:   0.10 GB 
# H2O cluster total cores:    2 
# H2O cluster allowed cores:  2

# Rtaing predicted - regression
cacao_url <- "http://coursera.h2o.ai/cacao.882.csv"
cacao <- h2o.importFile(path = cacao_url, destination_frame = "cacao")

#divided into train and test sets (crossvalidation will be used)
parts <- h2o.splitFrame(cacao, ratios = 0.8, seed=2019)
train <- parts[[1]]
test <- parts[[2]]

# STEP 2
summary(train)
glimpse(train) #rating is numeric value - "real" from 1 to 5
# set predicted column and predictors
x <- as.vector(colnames(cacao[,-7]))
y <- "Rating"

# STEP 3 : deep learning model
system.time(
  cacao_1 <- h2o.deeplearning(x, y, 
                              training_frame = train, 
                              nfold = 10,
                              seed = 2019,
                              model_id = "base_model"
  )
)
h2o.performance(cacao_1)

h2o.performance(cacao_1, newdata = test)

plot(cacao_1)


# STEP 4 : tuned model
system.time(cacao_2 <- h2o.deeplearning(x, y, 
                                        training_frame = train, 
                                        nfold = 8,
                                        epochs =12,
                                        stopping_rounds = 10,
                                        stopping_tolerance = 0, 
                                        stopping_metric = "RMSE",
                                        variable_importances = TRUE,
                                        l1 = 0,
                                        l2 = 0,
                                        seed = 2019,
                                        model_id = "tuned_model"
)
)
h2o.performance(cacao_2)
h2o.performance(cacao_2, newdata = test)
plot(cacao_2)

# STEP 5 : save model on local disc
path_to_save <- getwd()
h2o.saveModel(cacao_1, path=path_to_save)
h2o.saveModel(cacao_2, path=path_to_save)

#STEP6:  shutdown and save
h2o.shutdown()