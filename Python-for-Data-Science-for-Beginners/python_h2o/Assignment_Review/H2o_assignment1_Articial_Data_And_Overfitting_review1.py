#!/usr/bin/env python
# coding: utf-8

# ## 1. Create an artificial dataset

# In[59]:


import pandas as pd
import numpy as np
np.random.seed(1111)


# ### Creating the dataset structure

# I will create a dataset that represents the **Height (target)** of 1000 people along with the following features (**Weight,  Sex, Nationality, Sport, HealthyFood**)

# In[60]:


sex = ("Man", "Woman")


# In[61]:


nationality = ("Italian", "German", "Chinese" ,"American")


# In[62]:


sport = ("Basketball", "Handball", "Soccer", "Tennis")


# In[63]:


healthyfood = ("No", "Yes")


# In[64]:


data = pd.DataFrame({"Weight":np.random.randint(40,100,1000)})


# In[65]:


for i in range(len(data)):
    data.loc[i,"Sex"] = np.random.choice(sex)
    data.loc[i,"Nationality"] = np.random.choice(nationality)
    data.loc[i,"Sport"] = np.random.choice(sport)
    data.loc[i,"HealthyFood"] = np.random.choice(healthyfood)


# ### Creating the target value

# Now we have a dataset of 1000 people with their respective features, I will **create the target value "Height" following some rules:**
# 
# * **As a starting point, the Height will be created with the next formula:** *Height (in cm) = Weight (in kg) + 100.*
#     *  Before doing this, I will lower the Weight of woman to be more realistic.
# * **Next, I will add/subtract some Height based on the rules I've created.** For example:
#     * German people will tend to be taller while Chinese will be shorter.
#     * People who doesn't eat healthy will be shorter than people who eats healthy and has the same weight.
# * **Finally, I will distort some rows with unusual values (this will help in identifying the overfitting)**
# 
# 

# In[66]:


for i in range(len(data)):
    if data.loc[i,"Sex"] == "Woman":
        data.loc[i,"Weight"] -= 15
        
data["Height"] = data.Weight+100


# In[67]:


for i in range(len(data)):
    if data.loc[i,"Height"] > 220:
        data.loc[i,"Height"] = 220
    
    if data.loc[i,"Height"] < 140:
        data.loc[i,"Height"] = 140
    
    if data.loc[i,"Nationality"] == "German":
        data.loc[i,"Height"] += 10
    
    if data.loc[i,"Nationality"] == "Chinese":
        data.loc[i,"Height"] -= 15
    
    if data.loc[i,"Sport"] == "Basketball" or "Handball":
        data.loc[i,"Height"] += 15
    
    if data.loc[i,"HealthyFood"] == "No":
        data.loc[i,"Height"] -= 10


# In[68]:


distorted_rows = np.random.randint(0,999,100)
for i in distorted_rows:
     data.loc[i,"Height"] = np.random.randint(0,200)


# In[69]:


data.head()


# ## 2. Start h2o and upload data

# In[70]:


import h2o
h2o.init()


# In[71]:


data_h2o = h2o.H2OFrame(data)


# ## 3. Split data

# In[72]:


train, valid, test = data_h2o.split_frame(ratios=[0.8,0.1], destination_frames=["train", "valid", "test"])


# In[73]:


train.nrows, valid.nrows, test.nrows


# ## 4. First model (well-fitted)

# I will create a GBM model with its default values

# In[74]:


from h2o.estimators.gbm import H2OGradientBoostingEstimator


# In[75]:


m1 = H2OGradientBoostingEstimator()
m1.train(["Weight","Sex","Nationality","Sport","HealthyFood"],"Height", train, validation_frame=valid)


# In[76]:


m1.model_performance(train=True)


# In[77]:


m1.model_performance(valid=True)


# In[78]:


m1.model_performance(test)


# ## 5. Second model (over-fitted)

# In order to create and overfitted model, I will:
# * Increase the number of trees from 50 to 1000.
# * Increase the maximum depth of the trees from 5 to 10.

# In[79]:


m2 = H2OGradientBoostingEstimator(ntrees=1000, max_depth=10)
m2.train(["Weight","Sex","Nationality","Sport","HealthyFood"],"Height",train, validation_frame=valid)


# In[80]:


m2.model_performance(train=True)


# In[81]:


m2.model_performance(valid=True)


# In[82]:


m2.model_performance(test)


# ## 6. Comparison (m1: well-fitted; m2: over-fitted)

# In[83]:


print("Mean Absolute Error")
print("       m1     m2")
print("Train: %d --> %d" % (m1.mae(train=True), m2.mae(train=True)))
print("Test:  %d --> %d" % (m1.model_performance(test).mae(), m2.model_performance(test).mae()))


# Increasing the **number of trees** from 50 to **1000** and the **maximum tree depth** from 5 to **10** causes a clear overfitting. The **error on the test set increases** while the **error on the train set reduces**.

# #### Shut down the h2o cluster

# In[84]:


h2o.cluster().shutdown()
