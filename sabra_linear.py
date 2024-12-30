# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 15:56:52 2024

@author: m
"""

import pandas as pd
import os
#Get the data :
path =  'C:/Users/m/Desktop/centennial college/Introduction to AI 237402/linear assignment/'
filename = 'Ecom Expense.csv'
fullpath = os.path.join(path,filename)
ecom_exp_sabra = pd.read_csv(fullpath)
#Initial Exploration:
ecom_exp_sabra.head(3)
ecom_exp_sabra.shape
ecom_exp_sabra.columns.values
ecom_exp_sabra.dtypes
missing_values=ecom_exp_sabra.isnull().sum()
print(missing_values)
# Data transformation:
ecom_exp_sabra_transformed = pd.get_dummies(ecom_exp_sabra, drop_first=True)
categorical_columns = ecom_exp_sabra.select_dtypes(include=['object']).columns
columns_to_drop = list(categorical_columns) + ['Transaction ID']
ecom_exp_final = ecom_exp_sabra.drop(columns=columns_to_drop)
ecom_exp_final.head()
def normalize_dataframe(df):
    normalized_df = (df - df.min()) / (df.max() - df.min())
    return normalized_df
ecom_exp_normalized = normalize_dataframe(ecom_exp_final)
ecom_exp_normalized.head(2)

import matplotlib.pyplot as plt
ecom_exp_normalized.hist(figsize=(9, 10))
plt.show()

# Scatter matrix plot with alpha set to 0.4 and specified figure size
from pandas.plotting import scatter_matrix
ecom_exp_normalized.columns.values
selected_columns = ['Age ', ' Items ', 'Monthly Income', 'Transaction Time', 'Total Spend']
pd.plotting.scatter_matrix(ecom_exp_normalized[selected_columns], alpha=0.4, figsize=(13, 15), diagonal='kde')
plt.suptitle("Scatter Matrix of Selected Variables", fontsize=16)
plt.show()

# Build a model
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# i. Define the predictors and target variable (features and label)
X = ecom_exp_normalized.drop('Total Spend', axis=1)  # All columns except 'Total Spend'
y = ecom_exp_normalized['Total Spend']  # Target variable

# ii. Split the data into 65% training and 35% testing AND iii
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=28)  

# iv. Store the training and test data in the specified format
x_train_sabra = X_train
y_train_sabra = y_train
x_test_sabra = X_test
y_test_sabra = y_test

# v. Fit a linear regression model to the training data
model = LinearRegression()
model.fit(x_train_sabra, y_train_sabra)

# vi. Display the coefficients (weights of the model)
print("Model Coefficients:")
print(model.coef_)

# vii. Display the R^2 score of the model
train_score = model.score(x_train_sabra, y_train_sabra)
test_score = model.score(x_test_sabra, y_test_sabra)
print(f"Training R^2 Score: {train_score}")
print(f"Testing R^2 Score: {test_score}")

# viii. Add 'Record' to the list of predictors and rebuild the model
X_with_record = ecom_exp_normalized.drop(['Total Spend'], axis=1)  # Including 'Record'
X_train, X_test, y_train, y_test = train_test_split(X_with_record, y, test_size=0.35, random_state=42)

model_with_record = LinearRegression()
model_with_record.fit(X_train, y_train)

# ix. Display the coefficients for the new model
print("Model Coefficients with 'Record':")
print(model_with_record.coef_)

# x. Display the new model's R^2 score
train_score_with_record = model_with_record.score(X_train, y_train)
test_score_with_record = model_with_record.score(X_test, y_test)
print(f"Training R^2 Score with 'Record': {train_score_with_record}")
print(f"Testing R^2 Score with 'Record': {test_score_with_record}")