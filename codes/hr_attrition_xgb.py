# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 18:15:11 2019

@author: ashish_kumar2
"""

# This code is for building a XG Boost classifier model on the HR Attrition dataset

# Import Required packages and custom functions
import sys
import os
import pandas as pd
#import numpy as np
#import feather
#import pickle
#from sklearn.preprocessing import LabelEncoder
import joblib
#from sklearn.ensemble import RandomForestClassifier
# Import accuarcy metrices
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import recall_score, accuracy_score, f1_score
from sklearn.metrics import classification_report
#from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

# Import custom functions
sys.path.insert(0, './functions/')
#from ak_generic_fun import dup_col_rows, get_null, high_corr_sets, rem_high_corr
#from ak_plotting_fun import plt_numeric_categorical_cols, plt_corr_mat_heatmap
from ak_plotting_fun import roc_auc_curve_plot, label_and_plot_confusion_matrix

model_name='xgBoost'

inp_data_loc=r"D:\Test_Projects\HR_Attrition\output"

# Define input and output data location
out_data_loc= r"D:\Test_Projects\HR_Attrition\output"+'/'+ model_name


# Create a folder to store RF model data
if not os.path.exists(out_data_loc):
    os.makedirs(out_data_loc)


# Read Training and validation set data
model_dict=joblib.load(inp_data_loc+'/'+'model_dict')

x_train=model_dict['x_train']
y_train=model_dict['y_train']

x_valid=model_dict['x_valid']
y_valid=model_dict['y_valid']

######------------------Implement XG Boost Classifier-----------------########
from xgboost import XGBClassifier

# Build a support vector classifier
xgblassifier= XGBClassifier()

# Train classifier
attrition_xgblassifier_model=xgblassifier.fit(x_train,y_train)

# Predict on training set
y_train_pred_xgb= attrition_xgblassifier_model.predict(x_train)

# Predict on validation set
y_valid_pred_xgb= attrition_xgblassifier_model.predict(x_valid)


#------------------------Training set accuracy--------------------------------#

# Get Confuaion Matrix and plot it
xgb_confusion_matrix_train=label_and_plot_confusion_matrix(y_true=y_train,
                                                       y_pred=y_train_pred_xgb, 
                                                       test_type="Training",
                                                       model_name='XGB',
                                                       out_loc=out_data_loc)


# Get classification report which consist of Precision, Recall and F1-Score
# in pandas dataframe, In case the result is not required in pandas dataframe
# remove output_dict parameter (it will return a string)
xgb_classification_report_train=classification_report(y_true=y_train,
                                                     y_pred=y_train_pred_xgb,
                                                     digits=3,
                                                     output_dict=True)

# Get classification report in pandas dataframe
xgb_classification_report_train=pd.DataFrame(xgb_classification_report_train).transpose()
xgb_classification_report_train=xgb_classification_report_train.round(3)

# roc curve for Training set
roc_auc_curve_plot(y_true= y_train,
                   y_pred= y_train_pred_xgb,
                   classifier=attrition_xgblassifier_model,
                   test_type='Training',
                   model_name='XGB',
                   x_true= x_train, 
                   out_loc=out_data_loc)



#------------------------Validation set accuracy------------------------------#
# Get Confuaion Matrix and plot it
xgb_confusion_matrix_valid=label_and_plot_confusion_matrix(y_true=y_valid,
                                                       y_pred=y_valid_pred_xgb, 
                                                       test_type="Validation",
                                                       model_name='XGB',
                                                       out_loc=out_data_loc)


# Get classification report which consist of Precision, Recall and F1-Score
# in pandas dataframe, In case the result is not required in pandas dataframe
# remove output_dict parameter (it will return a string)
xgb_classification_report_valid=classification_report(y_true=y_valid,
                                                    y_pred=y_valid_pred_xgb, 
                                                    digits=3,
                                                    output_dict=True)

# Get classification report in pandas dataframe
xgb_classification_report_valid=pd.DataFrame(xgb_classification_report_valid).transpose()
xgb_classification_report_valid=xgb_classification_report_valid.round(3)



# roc_auc curve for Test set
roc_auc_curve_plot(y_true= y_valid,
                   y_pred= y_valid_pred_xgb,
                   classifier=attrition_xgblassifier_model,
                   test_type='Validation',
                   model_name='XGB',
                   x_true= x_valid,
                   out_loc=out_data_loc)