# Step One is to create a data set, as we saw in earlier videos. If you took my example and modified it, add a comment explaining what your modification was.
import datetime
import random
import pandas as pd
import numpy as np
now = datetime.datetime.now()
nfolds = 5
SEED = 123
random.seed(SEED)
N = 1000
bloodTypes = ['A','A','A','O','O','O','AB','B']
d = pd.DataFrame(list(range(1, N+1)), columns=['id'])
d['bloodTypes']=[bloodTypes[(i%len(bloodTypes))] for i in range(1, N+1)]
from numpy.random import randint
d['age'] = randint(18, 65, N)

from scipy.stats import truncnorm
def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

X = get_truncated_normal(mean=5, sd=2, low=0, upp=9)
d['healthyEating'] = X.rvs(N)
d['healthyEating'] = round(d['healthyEating'])
d['activeLifestyle'] = np.where(d['age']<30, 1, 0)
d['activeLifestyle'] = X.rvs(N)+d['activeLifestyle']
d['activeLifestyle'] = round(d['activeLifestyle'])
d['activeLifestyle'] = d['activeLifestyle'].clip(upper=9)
d['income'] = 20000 + (d['age']*3)**2
d['income'] = d['income'] + d['healthyEating']*500
d['income'] = d['income'] - d['activeLifestyle']*300
d['income'] = d['income'] + randint(0, 5000, N)
d['income'] = round(d['income'] / 100) * 100

# Step Two is to start h2o, and import your data.
print(d.describe())
import h2o
h2o.init()
people = h2o.H2OFrame(d, destination_frame = "people")

# Step Three is to split the data. If you plan to use cross-validation, split into train and test. Otherwise split into train, valid and test
train, valid, test = people.split_frame(ratios = [0.8, 0.1], destination_frames = ["train", "valid", "test"], seed=SEED)

# Step four is to choose either random forest or gbm, and make a model. It can be classification or regression. Then show the results, on both training data and the test data. You can show all the performance stats, or choose just one (e.g. I focused on MAE in the videos).
x = people.columns
y = "income"
x.remove(y)
from h2o.estimators.gbm import H2OGradientBoostingEstimator
mGBM_default = H2OGradientBoostingEstimator(model_id = "GBM_model_default_"+now.strftime("%Y-%m-%d_%H_%M_%S"),
                                   ntrees = 15,
                                   nfolds = nfolds,
                                   fold_assignment = "Modulo",
                                   keep_cross_validation_predictions = True,
                                   seed = SEED)
get_ipython().magic('time mGBM_default.train(x, y, train, validation_frame = valid)')
mGBM_default.plot()

print("GBM Default Train MAE score: %f" % mGBM_default.mae(train=True))
print("GBM Default Validation MAE score: %f" % mGBM_default.mae(valid=True))
print("GBM Default Cross Validation MAE score: %f" % mGBM_default.mae(xval=True))
perf = mGBM_default.model_performance(test)
print("GBM Default Test MAE score: %f" % perf.mae())

# Step five is then to try some alternative parameters, to build a different model, and show how the results differ.
mGBM_overfit = H2OGradientBoostingEstimator(model_id = "GBM_model_python_"+now.strftime("%Y-%m-%d_%H_%M_%S"),
                                   ntrees = 500,
                                   max_depth =10,
                                   nfolds = nfolds,
                                   fold_assignment = "Modulo",
                                   keep_cross_validation_predictions = True,
                                   seed = SEED)
get_ipython().magic('time mGBM_overfit.train(x, y, train, validation_frame = valid)')
mGBM_overfit.plot()
print("GBM Overfit Train MAE score: %f" % mGBM_overfit.mae(train=True))
print("GBM Overfit Validation MAE score: %f" % mGBM_overfit.mae(valid=True))
print("GBM Overfit Cross Validation MAE score: %f" % mGBM_overfit.mae(xval=True))
perf = mGBM_overfit.model_performance(test)
print("GBM Overfit Test MAE score: %f" % perf.mae())

# Step six is to save your script, then close down h2o, and run your script in a fresh session to make sure there are no bugs and the results are repeatable. If your script takes more than a couple of minutes to run, please put the required resources in a comment at the very top. (But first consider if you can get the same results with a smaller data set: this task is about tuning and over-fitting, not about big data.)
h2o.cluster().shutdown()
