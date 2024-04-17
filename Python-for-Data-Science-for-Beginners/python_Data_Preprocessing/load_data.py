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


success_alert = """
<div class="alert alert-success" role="alert">Loading data from %s was successful.</div>
"""

error_alert = """
<div class="alert alert-danger" role="alert">Error loading data from %s. %s</div>
"""


def load_data(widget):
    global csv_data
    path = wgt_file_location.value
    try:
        if wgt_header.value:
            csv_data = pd.read_csv(path, sep=wgt_separator.value)
        else:
            csv_data = pd.read_csv(path, sep=wgt_separator.value, names=wgt_manual_header.value.split(","))
        wgt_alert.value = success_alert % path
    except Exception as ex:
        print(ex)
        print(path)
        wgt_alert.value = error_alert % (path, ex)
    wgt_alert.visible = True
    

def preview_file(widget):
    path = wgt_file_location.value
    if path.startswith("http://") or path.startswith("https://") or path.startswith("ftp://"):
        raw_file = urllib.request.urlopen(path)
        wgt_file_preview.value = "<pre>%s</pre>" % raw_file.read(1000)
        raw_file.close()
    else:
        raw_file = open(path)
        wgt_file_preview.value = "<pre>%s</pre>" % raw_file.read(1000)
        raw_file.close()
    

def manual_columns(name,old,new):
    wgt_manual_header.visible = old
    
def update_path(name,old,new):
    wgt_file_location.value = new
    
    
def load_files(widget):
    files_list = {}
    root = os.curdir
    patterns = ["*.txt", "*.csv"]

    for path, subdirs, files in os.walk(root):
        for name in files:
            for pattern in patterns:
                if fnmatch(name, pattern):
                    files_list[os.path.join(path, name)] = os.path.join(path, name)
    widget.values = files_list

container = widgets.ContainerWidget()

wgt_alert = widgets.HTMLWidget()
wgt_file_location = widgets.TextWidget(description="Path/URL:")
wgt_file_path = widgets.DropdownWidget(description="Files List")
wgt_separator = widgets.TextWidget(description="Separator", value=",")
wgt_header = widgets.CheckboxWidget(description="First columns is a header?", value=True)
wgt_manual_header = widgets.TextWidget(description="Columns seperated by commas", visible=False)
wgt_load_data = widgets.ButtonWidget(description="Load Data")
wgt_preview_file = widgets.ButtonWidget(description="Preview File")
wgt_file_preview = widgets.HTMLWidget()

wgt_alert.visible = False

wgt_load_data.on_click(load_data)
wgt_preview_file.on_click(preview_file)
wgt_file_path.on_displayed(load_files)
wgt_file_path.on_trait_change(update_path, "value")
wgt_header.on_trait_change(manual_columns, "value")

container.children = (wgt_alert, wgt_file_path, wgt_file_location, wgt_separator, wgt_header, wgt_manual_header,
                      wgt_load_data, wgt_preview_file, wgt_file_preview)

display(container)
