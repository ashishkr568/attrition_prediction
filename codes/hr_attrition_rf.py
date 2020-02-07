# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 17:54:47 2019

@author: ashish_kumar2
"""

# This code is for building a random forest model on the HR Attrition dataset

# Import Required packages and custom functions
import sys
import os
import pandas as pd
#import numpy as np
#import feather
#import pickle
#from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.ensemble import RandomForestClassifier
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

model_name='Random_Forest'

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



#-------------------Hyperparameter tuning using GridSearchCV------------------#
# Create parameters for gridsearchcv
grid_param_rf={'n_estimators':[10,20,40,80,160,320,500,1000],
               'min_samples_leaf':[1,3,5,10,25],
               'max_features':[1,0.5,0.4,0.7],
               'bootstrap':[True,False]}


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


# Calculate the total number of runs
tot_runs_gris_search_rf= (len(grid_param_rf['n_estimators'])
                          * len(grid_param_rf['min_samples_leaf'])
                          * len(grid_param_rf['max_features'])
                          * len(grid_param_rf['bootstrap']))
print("Total Runs for Grid Search is - "+str(tot_runs_gris_search_rf))

# Build a dummy classifier
rf_classifier=RandomForestClassifier()

# Perform gridsearch for the above grid parameters
# We need to provide the desired scoring criteria and cv
rf_grid_search=GridSearchCV(estimator=rf_classifier,
                            param_grid=grid_param_rf,
                            scoring=custom_scorer,
                            cv=10,
                            n_jobs=4)

# Fit the above defined model with the training set, to get the best parameters 
rf_grid_search.fit(x_train,y_train)

# Get the best parameters and save it in a file
grid_cv_params_rf=rf_grid_search.best_params_
grid_cv_params_rf= pd.DataFrame(grid_cv_params_rf, index=[1,])
grid_cv_params_rf.to_csv(out_data_loc+'/'+"RF_Grid_serch_Params.csv", index=False)

#-----------------------------------------------------------------------------#



# Read Best parameters and test accuracy
grid_cv_params_rf=pd.read_csv(out_data_loc+"/"+"RF_Grid_serch_Params.csv")

# Build a random forest Regressor
rf_classifier= RandomForestClassifier(n_estimators = grid_cv_params_rf.n_estimators[0],
                                      min_samples_leaf = grid_cv_params_rf.min_samples_leaf[0],
                                      max_features = grid_cv_params_rf.max_features[0],
                                      bootstrap = grid_cv_params_rf.bootstrap[0],
                                      n_jobs=4)

# Train classifier
attrition_rf_model=rf_classifier.fit(x_train,y_train)

# Predict on training set
y_train_pred_rf= attrition_rf_model.predict(x_train)

# Predict on validation set
y_valid_pred_rf= attrition_rf_model.predict(x_valid)


# An employee is always a great asset to an organization. This tool will help
# help its HR team to check whether an employee is going to Resign or not.
# Once the probable employees who would resign are identified, the HR team can
# look for various options to retain them. The various options might be in 
# terms of compensation, good work life balance, travel options etc.


# True Positive - The tool correctly predicts that the employee is going to Resign
# False Positive- The tool predicts that the employee is going to resign which is not true
# False Negative- The tool predicts that the employee is not going to resign but resigns
# True Negative- The tool correctly predicts that the employee is not going to resign

# In the above case the cost of false negative is high. We need to limit the rate of 
# False negative, so a better accuracy metric will be the one wich deals with
# False Negative. We will consider Recall/ sensitivity, F1 Score,
# roc_auc score and finally auc score for model comparision.


#------------------------Training set accuracy--------------------------------#

# Get Confuaion Matrix and plot it
rf_confusion_matrix_train=label_and_plot_confusion_matrix(y_true=y_train,
                                                       y_pred=y_train_pred_rf, 
                                                       test_type="Training",
                                                       model_name='RF',
                                                       out_loc=out_data_loc)


# Get classification report which consist of Precision, Recall and F1-Score
# in pandas dataframe, In case the result is not required in pandas dataframe
# remove output_dict parameter (it will return a string)
rf_classification_report_train=classification_report(y_true=y_train,
                                                     y_pred=y_train_pred_rf,
                                                     digits=3,
                                                     output_dict=True)

# Get classification report in pandas dataframe
rf_classification_report_train=pd.DataFrame(rf_classification_report_train).transpose()
rf_classification_report_train=rf_classification_report_train.round(3)

# roc curve for Training set
roc_auc_curve_plot(y_true= y_train,
                   y_pred= y_train_pred_rf,
                   classifier=attrition_rf_model,
                   test_type='Training',
                   model_name='RF',
                   x_true= x_train, 
                   out_loc=out_data_loc)



#------------------------Validation set accuracy------------------------------#
# Get Confuaion Matrix and plot it
rf_confusion_matrix_valid=label_and_plot_confusion_matrix(y_true=y_valid,
                                                       y_pred=y_valid_pred_rf, 
                                                       test_type="Validation",
                                                       model_name='RF',
                                                       out_loc=out_data_loc)


# Get classification report which consist of Precision, Recall and F1-Score
# in pandas dataframe, In case the result is not required in pandas dataframe
# remove output_dict parameter (it will return a string)
rf_classification_repor_valid=classification_report(y_true=y_valid,
                                                    y_pred=y_valid_pred_rf, 
                                                    digits=3,
                                                    output_dict=True)

# Get classification report in pandas dataframe
rf_classification_repor_valid=pd.DataFrame(rf_classification_repor_valid).transpose()
rf_classification_repor_valid=rf_classification_repor_valid.round(3)



# roc_auc curve for Test set
roc_auc_curve_plot(y_true= y_valid,
                   y_pred= y_valid_pred_rf,
                   classifier=attrition_rf_model,
                   test_type='Validation',
                   model_name='RF',
                   x_true= x_valid, 
                   out_loc=out_data_loc)


# Get Feature importance in Random Forest
rf_imp_features=pd.DataFrame(attrition_rf_model.feature_importances_, 
                             index= x_train.columns).sort_values(0,ascending=False)

# Check top 10 important features
rf_imp_features.head(10)
