# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 10:53:45 2025

@author: m
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report

############################################################################################################
#############################################Exercice1##################################################
############################################################################################################

# Load and check the dataset
data_sabra=pd.read_csv("breast_cancer.csv")
print(data_sabra.info())
print("Display initial information")
print("Column Names & Types:\n", data_sabra.dtypes)
print("Missing Values:\n", data_sabra.isnull().sum())
print("Summary Statistics:\n", data_sabra.describe())

#####Pre-process and visualize the data#########

# Pre-process the data
data_sabra.replace('?',np.nan,inplace=True)
data_sabra['bare']=data_sabra['bare'].astype(float)
data_sabra.fillna(data_sabra.median(),inplace=True)
data_sabra.drop(columns=['ID'], inplace=True)

# Data Visualization
plt.figure(figsize=(12,6))
sns.histplot(data_sabra['thickness'], bins=10, kde=True)
plt.title("Distribution of Cell Thickness")
plt.show()

plt.figure(figsize=(12,6))
sns.boxplot(x='class',y='thickness', data=data_sabra,palette='Set1')
plt.title("Thickness Distribution by Class")
plt.show()

sns.pairplot(data_sabra,hue='class',palette='Set1')
plt.show()
             
plt.figure(figsize=(12,6))
sns.heatmap(data_sabra.corr(), annot=True, cmap='coolwarm',fmt='.2f')
plt.show()

sns.countplot(data=data_sabra,x='class',palette='Set2')
plt.title("Distribution of Classess")
plt.show()

###Training data
X=data_sabra.drop(columns=['class'])
y=data_sabra['class']

X_train, X_test, y_train , y_test = train_test_split(X,y, test_size=0.2,random_state=28)

##Build Classification Models

kernels = ['linear','rbf', 'poly', 'sigmoid']
results = {}
for kernel in kernels:
    C_value = 0.1 if kernel == 'linear' else 1.0 
    clf_sabra = SVC(kernel=kernel, C=C_value , random_state=28)
    clf_sabra.fit(X_train,y_train)
    # Predictions
    y_train_pred = clf_sabra.predict(X_train)
    y_test_pred = clf_sabra.predict(X_test)
    
    # Accuracy Scores
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    # Store results
    results[kernel] = {
       'classifier': clf_sabra,
       'train_accuracy': train_accuracy,
       'test_accuracy': test_accuracy,
       'confusion_matrix': conf_matrix
   }
   
    print(f"SVM with {kernel} kernel:")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy: {test_accuracy:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(classification_report(y_test, y_test_pred))
    print("--------------------------------------------------")
############################################################################################################
#############################################Exercice2##################################################
############################################################################################################
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import joblib

data_sabra_df2 = pd.read_csv('breast_cancer.csv')
data_sabra_df2['bare'] = data_sabra_df2['bare'].replace('?', np.nan).astype(float)
data_sabra_df2 = data_sabra_df2.drop(columns=['ID'])
X = data_sabra_df2.drop(columns=['class'])  
y = data_sabra_df2['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=28)
print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)

imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

num_pipe_sabra = Pipeline([
    ('imputer', imputer),  
    ('scaler', scaler)     
])

pipe_svm_sabra = Pipeline([
    ('num_pipe', num_pipe_sabra),  # Step 1: Preprocessing pipeline
    ('svm', SVC(random_state=28))  # Step 2: SVM classifier
])

print(num_pipe_sabra)

param_grid = {
    'svm__kernel': ['linear', 'rbf', 'poly'],  # Kernel types
    'svm__C': [0.01, 0.1, 1, 10, 100],        # Regularization parameter
    'svm__gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],  # Kernel coefficient for 'rbf' and 'poly'
    'svm__degree': [2, 3]                     # Degree of the polynomial kernel
}

print(param_grid)

grid_search_sabra = GridSearchCV(
    estimator=pipe_svm_sabra,  # Pipeline with preprocessing and SVM
    param_grid=param_grid,     # Grid search parameters
    scoring='accuracy',        # Scoring metric
    refit=True,                # Refit the best model on the entire dataset
    verbose=3                  # Verbosity level
)
print(grid_search_sabra)

grid_search_sabra.fit(X_train, y_train)

best_params = grid_search_sabra.best_params_
print("Best Parameters:", best_params)

best_estimator = grid_search_sabra.best_estimator_
print("Best Estimator:", best_estimator)
best_model_sabra = grid_search_sabra.best_estimator_
best_model_sabra.fit(X_train, y_train)
y_train_pred = best_model_sabra.predict(X_train)

# Calculate the accuracy score
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Training Accuracy:", train_accuracy)
joblib.dump(best_model_sabra, 'best_model_sabra.pkl')
joblib.dump(pipe_svm_sabra, 'pipe_svm_sabra.pkl')