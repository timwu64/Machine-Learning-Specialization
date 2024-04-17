# H2o Assignmern 2: Project Choice #1
# 
# http://coursera.h2o.ai/cacao.882.csv
# 
# This is a regression problem. You have to predict "Rating".
# 
# You should split the data into train, valid and test. Use a seed (of your choice) to make your experiments more reproducible. Alternatively, you can split into just train and test, and then use cross-validation.

# Step one is to start h2o, load your data set, and split it if necessary. By the end of this stage you should have three variables, pointing to three data frames on h2o: train, valid, test. However, if you are choosing to use cross-validation, you will only have two: train and test.

# In[1]:


import datetime
import random
import os
import warnings
warnings.filterwarnings("ignore")

nfolds = 5
SEED = 123
random.seed(SEED)
now = datetime.datetime.now()


# In[2]:


#Start H2o
import h2o
from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
h2o.init()


# In[3]:


#load data Set
df = h2o.import_file("http://coursera.h2o.ai/cacao.882.csv")


# In[5]:


df.summary()


# In[6]:


df = df.na_omit()
print(df.shape)


# Step two is to set x to be the list of columns you will use to train on, to be the column you will learn. Your choice of y decides if it will be a classification or a regression.

# In[7]:


train, valid, test = df.split_frame(ratios = [0.8, 0.1], destination_frames = ["train", "valid", "test"], seed=SEED)


# Step two is to set x to be the list of columns you will use to train on, to be the column you will learn. Your choice of y decides if it will be a classification or a regression.

# In[8]:


# Identify predictors and response
x = df.columns
y = "Rating"
x.remove(y)


# Step three is to create a baseline deep learning model. It is recommended to use all default settings (remembering to specify either nfolds or validation_frame); if you want to use different settings you should include a comment in your source file justifying this. Allowable reasons are that the default settings were taking too long on your hardware, or that they were over-fitting. Your script must include timing code.


from h2o.estimators.deeplearning import H2ODeepLearningEstimator
mDL = H2ODeepLearningEstimator(model_id = "DeepLearning_model_baseline_"+now.strftime("%Y-%m-%d_%H_%M_%S"),
                               seed = SEED
                              )
mDL.train(x, y, train, validation_frame = valid)



# In[17]:


print(mDL.model_performance(test))


# Step four is to produce a tuned model, that gives superior performance. However you should use no more than 10 times the running time of your baseline model, so again your script should be timing the model.

# For steps three and four you should include, in comments in your source file, the results in your chosen metric on the train, valid (or cross-validation) and test data sets, as well as the running time. Also include a comment at the top of your script describing how many cores and how much memory you used (you can get both of these from the output of h2o.init()).

#
#### Deep Learning Algorithm
#
#activation_opt = ["RectifierWithDropout",
#                  "TanhWithDropout"]
##L1 & L2 regularization
#l1_opt = [0, 0.00001,
#          0.0001,
#          0.001,
#          0.01,
#          0.1]
#
#l2_opt = [0, 0.00001,
#          0.0001,
#          0.001,
#          0.01,
#          0.1]
#
## Create the Hyperparameters
#dl_params = {
#             'activation': activation_opt,
#             "input_dropout_ratio" : [0,0.05, 0.1],  # input layer dropout ratio to improve generalization. Suggested values are 0.1 or 0.2.
#             'l1': l1_opt,
#             'l2': l2_opt,
#             'hidden_dropout_ratios':[[0.1,0.2,0.3], # hidden layer dropout ratio to improve generalization: one value per hidden layer.
#                                      [0.1,0.5,0.5],
#                                      [0.5,0.5,0.5]]
#             }
#
#search_criteria = {
#                   'strategy': 'RandomDiscrete',
#                   'max_runtime_secs': 1000,
#                   'seed':SEED
#                   }
#
## Prepare the grid object
#dl_grid = H2OGridSearch(model=H2ODeepLearningEstimator(
#                                                    epochs = 150,   ## hopefully converges earlier...
#                                                    adaptive_rate = True,  
#                                                    stopping_metric="AUTO",
#                                                    stopping_rounds=2,
#                                                    stopping_tolerance=0.0005,
#                                                    hidden=[200,200,200],      ## more hidden layers -> more complex interactions
#                                                    balance_classes= False,
#                                                    standardize = True,  # If enabled, automatically standardize the data (mean 0, variance 1). If disabled, the user must provide properly scaled input data.
#                                                    loss = "quantile",  # quantile for regression
#                                                    seed=SEED
#                                                    ),
#                        grid_id='dl_grid'+now.strftime("%Y-%m-%d_%H_%M_%S"),
#                        hyper_params=dl_params,
#                        search_criteria=search_criteria)
#
## Train the Model
#start = time.time() 
#dl_grid.train(x=x,y=y, 
#                training_frame=train,
#                validation_frame=valid
#                )
#
#end = time.time()
#(end - start)/60# Find the Model performance 
#dl_gridperf = dl_grid.get_grid(sort_by='RMSE',decreasing = False)
#dl_gridperf.sorted_metric_table()# Identify the best model generated with least error
#best_dl_model = dl_gridperf.models[0]
#best_dl_modelbest_dl_model.summary()['units']
#best_dl_model.plot()
#best_dl_model.model_performance(test) 
# Step five is to save both your models, to your local disk, and they should be submitted with your script. Use saveModel() (in R) or save_model() (Python), to export the binary version of the model. (Do not export a POJO.)

# In[40]:


mDL2 = H2ODeepLearningEstimator(model_id = "DeepLearning_model_best_"+now.strftime("%Y-%m-%d_%H_%M_%S"),
                                epochs = 80,   ## hopefully converges earlier...
                                score_validation_samples=10000,  # downsample validation set for faster scoring
                                score_duty_cycle=0.025,          # don't score more than 2.5% of the wall time
                                adaptive_rate=False,             # manually tuned learning rate
                                rate=0.01, 
                                rate_annealing=0.000002,            
                                momentum_start=0.2,              # manually tuned momentum
                                momentum_stable=0.4, 
                                momentum_ramp=10000000,  
                                stopping_metric="AUTO", # use early stopping to avoid overfitting
                                stopping_rounds=2,
                                stopping_tolerance=0.0005,
                                hidden=[200,200,200],      ## more hidden layers -> more complex interactions
                                balance_classes= False,
                                standardize = True,  # If enabled, automatically standardize the data (mean 0, variance 1). If disabled, the user must provide properly scaled input data.
                                loss = "quantile",  # quantile for regression
                                seed=SEED,
                                activation = "RectifierWithDropout",
                                hidden_dropout_ratios = [0.1, 0.5, 0.5], 
                                input_dropout_ratio = 0.05,
                                l1=1.0E-5,                      
                                l2=0.01
                               )

mDL2.train(x, y, training_frame=train,validation_frame=valid)


mDL2.model_performance(valid) 

print(mDL2.model_performance(test))

cwd = os.getcwd()

h2o.save_model(mDL, path = cwd + "/H2o_model_Base_deep_learning")
h2o.save_model(mDL2, path = cwd + "/H2o_model_BEST_deep_learning")


# Step six is to save your script, shutdown h2o, and run your script again in a fresh session to be sure the results are reproducible, and that there are no bugs. This is a very important step: scripts that do not run might fail to be graded.
#Additional Guidelines:
#
#Also include some justification that your tuned model has not over-fitted. If you have used early stopping, pointing that out should be sufficient.
#
#Make sure your script is clear, easy to understand for other students, and no longer than it needs to be. (It is unlikely to need to be more than 60 lines.)
#
#You may use a grid search to discover good parameters. However do not include it in your final script.
#
#You may use AutoML to get ideas; however do not include it in your final script, and remember you must use a single deep learning model: no other algorithms, no ensembles.


h2o.cluster().shutdown()

