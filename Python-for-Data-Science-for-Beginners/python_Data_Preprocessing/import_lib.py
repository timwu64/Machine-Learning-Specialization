# Core Libraries
import os
from fnmatch import fnmatch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.html import widgets
from IPython.html.widgets import interact

from IPython.display import display
import urllib.request, urllib.parse, urllib.error
from datetime import datetime

# Feature Extraction
from sklearn.feature_extraction import DictVectorizer

# Preprocessing
from sklearn import preprocessing

# External
from sklearn.externals import joblib

# Hide Warnings
import warnings
warnings.filterwarnings('ignore')

# Configure Pandas
pd.options.display.max_columns = 100
pd.options.display.width = 120
