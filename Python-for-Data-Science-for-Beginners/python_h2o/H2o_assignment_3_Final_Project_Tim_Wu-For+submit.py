
# coding: utf-8

# Final Project for Tim Wu
# 
# Step one is to start h2o, load your chosen data set(s) and follow the project-specific data manipulation steps.
# 
# This is a regression problem. The model will predict "price". The goal is reach the target RMSE below 123,000

# In[1]:


## Import Libraries

import h2o
from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator
import warnings
warnings.filterwarnings("ignore")
SEED = 123


# In[2]:


#start h2o
h2o.init()


# In[3]:


# load your chosen data set(s)
df = h2o.import_file("http://coursera.h2o.ai/house_data.3487.csv")


# In[4]:


print(df.shape)


# Split date into year and month columns. Then combine them into a numeric date column

# In[5]:


df["date"] = df["date"].gsub(pattern = "T000000", replacement = "")
df["date"] = df["date"].trim()


# In[6]:


df_month_date = df["date"].strsplit(pattern = "20\d\d")
df_month = df_month_date[1].strsplit(pattern = "\d\d$")
df_month.set_names(['month'])


# In[7]:


df_year = df["date"].strsplit(pattern = "\d\d\d\d$")
df_year.set_names(['year'])


# In[8]:


df = df.cbind(df_year)
df = df.cbind(df_month)


# In[9]:


df["year"] = df["year"].asnumeric()
df["month"] = df["month"].asnumeric()
df["date"] = df["date"].asnumeric()


# In[10]:


df.summary()


# Split the data into train and test, using 0.9 for the ratio, and a seed of 123. That should give 19,462 training rows and 2,151 test rows.

# In[11]:


# Split the train dataset
train, test = df.split_frame(ratios=[0.9], seed=SEED)


# In[12]:


print(train.shape)
print(test.shape)


# In[13]:


train, valid = df.split_frame(ratios=[0.8], seed=SEED)


# In[14]:


# Seperate the target data and store it into y variable
y = 'price'


# In[15]:


# remove target and Id column from the dataset and store rest of the columns in X variable
X = list(train.columns)
X.remove(y)
X.remove('id')


# At the end of this step you will have `train`, `test`, `x` and `y` variables, and possibly `valid` also. Check you have the correct number of rows and columns (as specified in the project description) before moving on.

# In[16]:


### 1. Gradient Boosting Machine (GBM)

mGBM = H2OGradientBoostingEstimator(model_id = "GBM_model_baseline",
                                   stopping_rounds = 4,
                                   stopping_metric ="RMSE",
                                   stopping_tolerance = 0.001,
                                   ntrees = 200,
                                   max_depth =5,
                                   nfolds = 5,
                                   fold_assignment = "Modulo",
                                   keep_cross_validation_predictions = True,
                                   seed = SEED)
mGBM.train(X, y, train, validation_frame = valid)
 
h2o.save_model(mGBM, "./models/")


# In[17]:


print(mGBM.model_performance(test))


# The baseline_gbm_model gives the RMSE: 125,373 which is higher than the target RMSE 123,000

# In[18]:


### 2. Random Forest Algorithm

mRF = H2ORandomForestEstimator(model_id = "RF_model_baseline",
                               nfolds = 5,
                               fold_assignment = "Modulo",
                               keep_cross_validation_predictions = True,
                               seed = SEED
                              )
mRF.train(X, y, train, validation_frame = valid)
 
h2o.save_model(mRF, "./models/")


# In[19]:


print(mRF.model_performance(test))


# The baseline_rf_model gives the RMSE: 128,995 which is higher than the target RMSE 123,000 and wrose than GBM baseline

# In[20]:


from h2o.estimators.deeplearning import H2ODeepLearningEstimator
mDL = H2ODeepLearningEstimator(model_id = "DeepLearning_model_baseline",
                               nfolds = 5,
                               fold_assignment = "Modulo",
                               keep_cross_validation_predictions = True,
                               seed = SEED
                              )
mDL.train(X, y, train, validation_frame = valid)
 
h2o.save_model(mDL, "./models/")


# In[21]:


print(mDL.model_performance(test))


# The baseline_dl_model gives the RMSE: 127,588 which is higher than the target RMSE 123,000 and wrose than GBM baseline

# Step three is to train a stacked ensemble of the models I made in step two.

# In[23]:


models = [mRF.model_id, mGBM.model_id, mDL.model_id]
 
mSE = H2OStackedEnsembleEstimator(model_id = "SE_model",
                                 base_models=models)
mSE.train(X, y, train, validation_frame = valid)
 
h2o.save_model(mSE, "./models/")


# In[24]:


print(mSE.model_performance(test))


# The stackedensemble gives the RMSE: 338,215 which is higher than the target RMSE 123,000 and wrose than GBM baseline

# Repeat steps two and three until your best model (which is usually your ensemble model, but does not have to be) has the minimum required performance on the validation data.

# In[25]:


### Gradient Boosting Machine (GBM) after grid search

mGBM_best = H2OGradientBoostingEstimator(model_id = "GBM_model_best_4",
                                   score_tree_interval=5,     # For early stopping
                                   stopping_rounds=3,         # For early stopping
                                   stopping_tolerance=0.0005,
                                   ntrees=200,
                                   col_sample_rate = 0.2,
                                   learn_rate = 0.1,
                                   sample_rate = 0.8,
                                   seed = SEED)
mGBM_best.train(X, y,
                training_frame=train,
                validation_frame=valid)

h2o.save_model(mGBM_best, "./models/")


# Step four is to get the performance on the test data of the chosen model (GBM), and confirm that this also reaches the minimum target on the test data. Record your model performance in comments at the end of your script

# In[26]:


print(mGBM_best.model_performance(test))


# After grid search, best_gbm_model gives the best RMSE: 119,155 which is lower than the target RMSE below 123,000

# In[27]:


h2o.cluster().shutdown()

