# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 17:45:47 2019

@author: ashish_kumar2
"""

# This code is for building a SVM model on the HR Attrition dataset

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

model_name='SVM'

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


################--------SVM Classifier Implementation------####################

#------------------Hyperplane tuning using GridSearch-------------------------#
# Create parameters for gridsearchcv
grid_param_svc=[{'kernel':['rbf'],
               'C':[1,10,100,1000],
               'gamma':[1e-3,1e-4]},
                {'kernel':['linear'],
                 'C':[1,10,100,1000],
                 'gamma':[1e-3,1e-4]},
                 {'kernel':['sigmoid'],
                  'C':[1,10,100,1000],
                  'gamma':[1e-3,1e-4]},
                 {'kernel':['poly'],
                  'C':[1,10,100,1000],
                  'gamma':[1e-3,1e-4], 
                  'degree':[1,2,3]}]
    

#grid_param_rf={'n_estimators':[10,20,],
#               'min_samples_leaf':[1],
#               'max_features':[1],
#               'bootstrap':[True,False]}

# Create a custom scoring function. 
# We will be using recall for the positive class.
def custom_recall_fun(y,y_pred):
    confusion_mat_temp=confusion_matrix(y,y_pred, labels=[1,0]).T
    recall_positive=confusion_mat_temp[0,0]/(confusion_mat_temp[0,0]+confusion_mat_temp[1,0])
    return recall_positive

# Use make scorer from sklearn.metrices to create a scoring function
custom_scorer = make_scorer(score_func=custom_recall_fun, greater_is_better=True)


# Build a dummy classifier
from sklearn.svm import SVC
svc_classifier=SVC()

# Perform gridsearch for the above grid parameters
# We need to provide the desired scoring criteria and cv
svc_grid_search=GridSearchCV(estimator=svc_classifier,
                            param_grid=grid_param_svc,
                            scoring=custom_scorer,
                            cv=10,
                            n_jobs=4)

# Fit the above defined model with the training set, to get the best parameters 
svc_grid_search.fit(x_train,y_train)

# Get the best parameters and save it in a file
grid_cv_params_svc=svc_grid_search.best_params_
grid_cv_params_svc= pd.DataFrame(grid_cv_params_svc, index=[1,])
grid_cv_params_svc.to_csv(out_data_loc+'/'+"svc_Grid_serch_Params.csv", index=False)




# Read Best parameters and test accuracy
grid_cv_params_svc=pd.read_csv(out_data_loc+"/"+"svc_Grid_serch_Params.csv")

# Import Packages
from sklearn.svm import SVC

# Build a support vector classifier
svclassifier= SVC(kernel = grid_cv_params_svc.kernel[0],
                           gamma = grid_cv_params_svc.gamma[0],
                           C = grid_cv_params_svc.C[0],
                           probability=True)

# Train classifier
attrition_svclassifier_model=svclassifier.fit(x_train,y_train)

# Predict on training set
y_train_pred_svc= attrition_svclassifier_model.predict(x_train)

# Predict on validation set
y_valid_pred_svc= attrition_svclassifier_model.predict(x_valid)


#------------------------Training set accuracy--------------------------------#

# Get Confuaion Matrix and plot it
svc_confusion_matrix_train=label_and_plot_confusion_matrix(y_true=y_train,
                                                       y_pred=y_train_pred_svc, 
                                                       test_type="Training",
                                                       model_name='SVC',
                                                       out_loc=out_data_loc)


# Get classification report which consist of Precision, Recall and F1-Score
# in pandas dataframe, In case the result is not required in pandas dataframe
# remove output_dict parameter (it will return a string)
svc_classification_report_train=classification_report(y_true=y_train,
                                                     y_pred=y_train_pred_svc,
                                                     digits=3,
                                                     output_dict=True)

# Get classification report in pandas dataframe
svc_classification_report_train=pd.DataFrame(svc_classification_report_train).transpose()
svc_classification_report_train=svc_classification_report_train.round(3)

# roc curve for Training set
roc_auc_curve_plot(y_true= y_train,
                   y_pred= y_train_pred_svc,
                   classifier=attrition_svclassifier_model,
                   test_type='Training',
                   model_name='SVC',
                   x_true= x_train, 
                   out_loc=out_data_loc)



#------------------------Validation set accuracy------------------------------#
# Get Confuaion Matrix and plot it
svc_confusion_matrix_valid=label_and_plot_confusion_matrix(y_true=y_valid,
                                                       y_pred=y_valid_pred_svc, 
                                                       test_type="Validation",
                                                       model_name='SVC',
                                                       out_loc=out_data_loc)


# Get classification report which consist of Precision, Recall and F1-Score
# in pandas dataframe, In case the result is not required in pandas dataframe
# remove output_dict parameter (it will return a string)
svc_classification_report_valid=classification_report(y_true=y_valid,
                                                    y_pred=y_valid_pred_svc, 
                                                    digits=3,
                                                    output_dict=True)

# Get classification report in pandas dataframe
svc_classification_report_valid=pd.DataFrame(svc_classification_report_valid).transpose()
svc_classification_report_valid=svc_classification_report_valid.round(3)



# roc_auc curve for Test set
roc_auc_curve_plot(y_true= y_valid,
                   y_pred= y_valid_pred_svc,
                   classifier=attrition_svclassifier_model,
                   test_type='Validation',
                   model_name='SVC',
                   x_true= x_valid,
                   out_loc=out_data_loc)