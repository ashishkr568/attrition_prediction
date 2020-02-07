# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 15:40:07 2019

@author: ashish_kumar2
"""

# This python file id used to preprocess the HR Attrition dataset, It performs
# the following tasks
# * Import required packages
# * Read the dataset from the predifined folder
# * Keep a blind dataset as test set
# * Check for null values
# * Check for duplicate values
# * Plot data for visualization
# * Check and remove data with 0 variablity
# * Encode data using label encoder and create dummy variables for categorical data
# * Get data variablity and remove highly correlated values
# * Split data into training and validation set
# * Scale training and validation set data
# * As there is huge class imbalance, Use SMOTE to create synthetic data (Oversampling)
# * Save required fields


# The below fields are being saved in a the mentioned file format
# * Preprocessed data in the name of model_data (feather format)
# * Test set (feather format)
# * Label encoder used for encoding Attrition in pickle format




# import packages
import sys
import pandas as pd
import numpy as np
import feather
import pickle
from sklearn.preprocessing import LabelEncoder
import joblib

# Import custom functions
sys.path.insert(0, './functions/')
from ak_generic_fun import dup_col_rows, get_null, high_corr_sets, rem_high_corr
from ak_plotting_fun import plt_numeric_categorical_cols, plt_corr_mat_heatmap
from ak_plotting_fun import roc_auc_curve_plot, label_and_plot_confusion_matrix

# Set seed for reproducablity
np.random.seed(42)


# Define input and output data location
inp_data_loc= r"D:\Test_Projects\HR_Attrition\ibm_hr_analytics_attrition_dataset"
out_data_loc= r"D:\Test_Projects\HR_Attrition\output"

# Input file name
f_name= r"WA_Fn-UseC_-HR-Employee-Attrition.csv"


# Read data
inp_ds=pd.read_csv(inp_data_loc+"/"+f_name)


# Take 10 % of the data at random for test set from the input dataset
test_set_ds=inp_ds.sample(frac=0.1)

# Remove testset for the input dataset and reset index
inp_ds=inp_ds[~inp_ds.index.isin(test_set_ds.index)]

    
# Check for Null values
null_info=get_null(inp_ds)

# Suppose there are any missing values and we need to do imputation for it, in
# that case we need to check for categorical and numerical values and then 
# impute the numerical values with the the mean, mode or median and in case of 
# categorical valued we van impute it with mode/ most frequent terms.

# Check for duplicated rows and columns
inp_ds= dup_col_rows(inp_ds)



# Data Visualization
# plot all the numeric and categorical variables
plt_numeric_categorical_cols(ds=inp_ds, out_loc=out_data_loc)



# Findings (after analysing the plots)
# There are no variablity in the following parameters
# 1. Over18
# 2. EmployeeCount
# 3. StandardHours

# We also need to remove the following parameter as they just for reference
# EmployeeNumber

# List of columns to be removed
rem_cols=["Over18","EmployeeCount","StandardHours","EmployeeNumber"]


# Remove unnecessary columns from the input dataset
inp_ds=inp_ds.drop(rem_cols,axis=1)




# Convert Categorical variables to Numeric variables
# We need to label encode the output/deoendent variable
le = LabelEncoder()
# Fit the Label encoder for dependent variable, we will use the same for encoding the test set
le_att=le.fit(inp_ds.Attrition)
# Tranform the dependent variable 
inp_ds.Attrition=le_att.transform(inp_ds.Attrition)



# Now we need to convert independent categorical variables to numerical variables
# We will be creating dummy variables for all the categories using pandas get dummies framework

# Get the list of categorical variables and create dummy variables for each of them
cat_cols=list(inp_ds.select_dtypes(include=['object']).columns)
inp_ds_dummy=pd.get_dummies(inp_ds, columns=cat_cols)


# Get variablity of the dataset using standard deviation
inp_ds_variablity=round(inp_ds_dummy.std(),2).sort_values()

# Check if the data is skewed
attrition_count=inp_ds_dummy.Attrition.value_counts()
skewness=round(attrition_count[0]/attrition_count[1],2)

if(.80<skewness<1.2):
    print("There is no class imbalance.")
else:
    print("The data is skewed and the value of skewness is - "+str(skewness))
# The value of skewness should be close to 1. In this case its value is quite 
# far from 1, hence the data is highly skewed and we need to deal with class
# imbalance problem



# Plot heatmap to check for correlation
plt_corr_mat_heatmap(inp_ds_dummy, out_data_loc)


# As the number of parameters are more, visual technique is not very effecient
# We need to get the generate sets of parameters based on correlation coeffecient 
# and then remove those highly correlated parameters

# Get highly Correlated sets based on passed correlation coeffecient
corr_sets= high_corr_sets(ds=inp_ds_dummy, corr_coef=0.9)


# Remove highly correlated values form the data to be passed for model development
inp_ds_wo_corr= rem_high_corr(ds=inp_ds_dummy, corr_coef=0.9)


# Data used for Model development
model_ds=inp_ds_wo_corr

# Seperate dependent and independent variables and split them into training and validation set
# Dependent variable name
dep_var_name="Attrition"

x,y=model_ds.drop(dep_var_name, axis=1), model_ds[dep_var_name]

# Split data into training and validation set
from sklearn.model_selection import train_test_split
x_train,x_valid,y_train,y_valid=train_test_split(x,y, test_size=0.10, stratify=y)

# Scale all independent varaibles
from sklearn.preprocessing import StandardScaler
# Fit scalar on training set
std_scalar= StandardScaler().fit(x_train)

# Use this scalar to transform training and validation set
train_cols=x_train.columns
x_train=std_scalar.transform(x_train)
x_train=pd.DataFrame(x_train, columns=train_cols)

valid_cols=x_valid.columns
x_valid=std_scalar.transform(x_valid)
x_valid=pd.DataFrame(x_valid, columns=valid_cols)
# As there is huge class imblalance, lets try to deal it using SMOTE
# SMOTE - Synthetic Minority Over Sampling Techniques.
# There are following other ways as well to deal with severe class imbalance
# 1. Synthesis of new minority class instance
# 2. Over sampling of minority class
# 3. Under Sampling of majority class
# 4. tweek the cost function to make misclassification of minority instances 
#    more important than misclassification of majority instances

#from imblearn.over_sampling import SMOTE
#smt=SMOTE()
#x_train,y_train = smt.fit_sample(x_train,y_train)


# Create a dictionary for consisting of training and validation set
model_dict={'x_train':x_train,
            'y_train':y_train,
            'x_valid':x_valid,
            'y_valid':y_valid}

# Save the training and validation set
joblib.dump(model_dict,out_data_loc+'/'+'model_dict')



# Save scaling used in joblib file format
joblib.dump(std_scalar,out_data_loc+'/'+'std_scalar')

# Save the label encoder to pickle file in order to use it for the test set
# Save the Attrition Encoder
pkl_output = open(out_data_loc+'/'+'Attrition_encoder.pkl', 'wb')
pickle.dump(le_att, pkl_output)
pkl_output.close()



## Save the cleaned dataframe in feather file format for faster read and write
#inp_ds_wo_corr.reset_index(inplace=True) # Feather needs to reset index
#inp_ds_wo_corr.to_feather(out_data_loc+'/'+'model_data')
