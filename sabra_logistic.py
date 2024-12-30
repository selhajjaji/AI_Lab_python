# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 09:42:24 2024

@author: m
"""
import pandas as pd
import os

###part A
# Load the Titanic dataset and name it with my first name
path =  'C:/Users/m/Desktop/centennial college/Introduction to AI 237402/logisitic regression/LAB4'
filename = 'titanic.csv'
fullpath = os.path.join(path,filename)
titanic_sabra = pd.read_csv(fullpath)

###part B initial Exploration
#Display the first 3 records
print(titanic_sabra.head(3))
#Display the shape of the dataframe
print("Shape of the DataFrame:", titanic_sabra.shape)
#Display the names, types, and counts of missing values per column
print(titanic_sabra.info())
#Display unique values for 'Sex' and 'Pclass'
print("Unique values in 'Sex':", titanic_sabra['Sex'].unique())
print("Unique values in 'Pclass':", titanic_sabra['Pclass'].unique())

###part C Data Visualization
import matplotlib.pyplot as plt

# Step c.1.a: Bar chart showing # of survived versus passenger class
pclass_survived = pd.crosstab(titanic_sabra['Pclass'], titanic_sabra['Survived'])
pclass_survived.plot(kind='bar', stacked=True, color=['red', 'blue'])
plt.title("Survival Count by Passenger Class -Sabra")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
plt.legend(["Did Not Survive", "Survived"])
plt.show()
# Step c.1.b: Bar chart showing # of survived versus gender
sex_survived = pd.crosstab(titanic_sabra['Sex'], titanic_sabra['Survived'])
sex_survived.plot(kind='bar', stacked=True, color=['purple', 'green'])
plt.title("Survival Count by Gender - Sabra")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")
plt.legend(["Did Not Survive", "Survived"])
plt.show()
# Step c.2: Scatter matrix to analyze relationships between selected attributes
from pandas.plotting import scatter_matrix

selected_attributes = titanic_sabra[['Survived','Sex', 'Pclass', 'Fare', 'SibSp', 'Parch']]
scatter_matrix(selected_attributes, figsize=(10, 10), diagonal='kde', color='blue')
plt.suptitle("Scatter Matrix for Titanic Dataset - Sabra")
plt.show()

###part d Data Transformation
#Drop the columns identified in part b.4
titanic_sabra = titanic_sabra.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
#Use get_dummies to convert categorical variables into numeric
dummies = pd.get_dummies(titanic_sabra[['Sex', 'Embarked']])
# Attach the newly created variables to dataframe and drop the original columns.
titanic_sabra = pd.concat([titanic_sabra, dummies], axis=1)
#Remove the original categorical variables columns. 
titanic_sabra.drop(columns=['Sex', 'Embarked'], inplace=True)
#Replace missing values in "Age" with the mean of the column
titanic_sabra['Age'] = titanic_sabra['Age'].fillna(titanic_sabra['Age'].mean())
# Change all column types to float
titanic_sabra = titanic_sabra.astype('float')
# Question 7 call the method info()
print(titanic_sabra.info())

#Function that accepts a dataframe as an argument and normalizes 
#all the data points in the dataframe
def normalize_dataframe(df):
     # Create a copy of the DataFrame to avoid modifying the original
    normalized_df = df.copy()    
    # Normalize each column using the Min-Max formula
    for column in normalized_df.columns:
        min_value = normalized_df[column].min()
        max_value = normalized_df[column].max()
        normalized_df[column] = (normalized_df[column] - min_value) / (max_value - min_value)
    
    return normalized_df
# Calling the normalize function that Normalize the transformed DataFrame
normalized_titanic = normalize_dataframe(titanic_sabra)
# Display the first two records of the normalized DataFrame
print(normalized_titanic.head(2))
# Generate histograms for all variables
normalized_titanic.hist(figsize=(9, 10))
plt.suptitle('Histograms of All Variables - Sabra')
plt.show()
# Assuming 'Survived' is the target variable
x_sabra = normalized_titanic.drop(columns=['Survived'])
y_sabra = normalized_titanic['Survived']
from sklearn.model_selection import train_test_split
# Set the random seed 
random_seed = 28 
# Split the data into training and testing sets (70% train, 30% test)
x_train_sabra, x_test_sabra, y_train_sabra, y_test_sabra = train_test_split(
    x_sabra, y_sabra, test_size=0.3, random_state=random_seed)

from sklearn.linear_model import LogisticRegression
# Initialize the model
sabra_model = LogisticRegression()
# Fit the model to the training data
sabra_model.fit(x_train_sabra, y_train_sabra)
import numpy as np
# Create a DataFrame to display the coefficients
coef_df = pd.DataFrame(zip(x_train_sabra.columns, np.transpose(sabra_model.coef_)))
coef_df.columns = ['Feature', 'Coefficient']
print(coef_df)

from sklearn.model_selection import cross_val_score
# Set up test sizes from 10% to 50% in increments of 5%
test_sizes = np.arange(0.1, 0.55, 0.05)
# Store the results for each test size
results = []

# Loop through each test size
for test_size in test_sizes:
    # Split the data based on the current test size
    x_train, x_test, y_train, y_test = train_test_split(
        x_sabra, y_sabra, test_size=test_size, random_state=random_seed
    )
    
    # Fit the model   
    sabra_model.fit(x_train, y_train)
    
    # Validate the model using cross_val_score
    scores = cross_val_score(sabra_model, x_train, y_train, cv=10)
    
    # Append the min, mean, and max scores to results
    results.append((test_size, scores.min(), scores.mean(), scores.max()))

# Create a DataFrame for better visualization of results
results_df = pd.DataFrame(results, columns=['Test Size', 'Min Accuracy', 'Mean Accuracy', 'Max Accuracy'])
print(results_df)

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Step 1: Rebuild the model using the 70% - 30% train/test split
model = LogisticRegression()
model.fit(x_train_sabra, y_train_sabra)

# Step 2: Define the new variable to store predicted probabilities
y_pred_sabra = model.predict_proba(x_test_sabra)

# Step 3: Define the another variable with a threshold of 0.5
y_pred_sabra_flag = y_pred_sabra[:, 1] > 0.5

# Step 4: Calculate metrics for the threshold of 0.5
accuracy_05 = accuracy_score(y_test_sabra, y_pred_sabra_flag)
conf_matrix_05 = confusion_matrix(y_test_sabra, y_pred_sabra_flag)
classification_report_05 = classification_report(y_test_sabra, y_pred_sabra_flag)

# Print results for threshold 0.5
print("Accuracy (threshold 0.5):", accuracy_05)
print("Confusion Matrix (threshold 0.5):\n", conf_matrix_05)
print("Classification Report (threshold 0.5):\n", classification_report_05)

# Step 9: Change threshold to 0.75 and calculate metrics
y_pred_sabra_flag_075 = y_pred_sabra[:, 1] > 0.75
accuracy_075 = accuracy_score(y_test_sabra, y_pred_sabra_flag_075)
conf_matrix_075 = confusion_matrix(y_test_sabra, y_pred_sabra_flag_075)
classification_report_075 = classification_report(y_test_sabra, y_pred_sabra_flag_075)

# Print results for threshold 0.75
print("\nAccuracy (threshold 0.75):", accuracy_075)
print("Confusion Matrix (threshold 0.75):\n", conf_matrix_075)
print("Classification Report (threshold 0.75):\n", classification_report_075)

