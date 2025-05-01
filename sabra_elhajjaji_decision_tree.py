# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 14:52:57 2025

@author: m
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import joblib
import graphviz

# Load and check the dataset
path='D:/Centennial College-Winter2025/Supervised Learning/decision_tree_assignment/'
filename='student-por.csv'
fullpath=os.path.join(path,filename)
data_sabra=pd.read_csv(fullpath,delimiter=';')
print(data_sabra.info())
print("Display initial information")
print("Column Names & Types:\n", data_sabra.dtypes)
print("Missing Values:\n", data_sabra.isnull().sum())
print("Summary Statistics:\n", data_sabra.describe())
print("Categorical values:\n",data_sabra.select_dtypes(include='object').nunique())

data_sabra['pass_sabra']= ((data_sabra['G1']+data_sabra['G2']+data_sabra['G3']) >= 35).astype(int)
data_sabra.drop(columns=['G1','G2','G3'],inplace=True)
target_variable_sabra=data_sabra['pass_sabra']
features_sabra=data_sabra.drop(columns='pass_sabra',axis=1)
print(features_sabra.value_counts())
print(target_variable_sabra.value_counts())
# Identify numeric and categorical features
numeric_features_sabra = features_sabra.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features_sabra = features_sabra.select_dtypes(include=['object']).columns.tolist()

# Column transformer for categorical encoding
transformer_sabra = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), cat_features_sabra)],
    remainder='passthrough'
)

# Decision Tree classifier
clf_sabra = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=28)

# Pipeline
pipeline_sabra = Pipeline(steps=[
    ('transformer', transformer_sabra),
    ('classifier', clf_sabra)
])

# Train-test split
X_train_sabra, X_test_sabra, y_train_sabra, y_test_sabra = train_test_split(
    features_sabra, target_variable_sabra, test_size=0.2, random_state=28
)

# Fit model
pipeline_sabra.fit(X_train_sabra, y_train_sabra)

# Cross-validation
from sklearn.model_selection import KFold

cv = KFold(n_splits=10, shuffle=True, random_state=28)
cv_scores = cross_val_score(pipeline_sabra, X_train_sabra, y_train_sabra, cv=cv)
print("Cross-validation scores:", cv_scores)
print("Mean CV score:", cv_scores.mean())

# Visualize the tree
tree_model = pipeline_sabra.named_steps['classifier']
feature_names = pipeline_sabra.named_steps['transformer'].get_feature_names_out()
dot_data = export_graphviz(tree_model, out_file=None,
                           feature_names=feature_names,
                           class_names=['Fail', 'Pass'],
                           filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("decision_tree_sabra", format='png')

# Accuracy on train/test
train_accuracy = accuracy_score(y_train_sabra, pipeline_sabra.predict(X_train_sabra))
test_accuracy = accuracy_score(y_test_sabra, pipeline_sabra.predict(X_test_sabra))
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

# Metrics
y_pred = pipeline_sabra.predict(X_test_sabra)
print("Precision:", precision_score(y_test_sabra, y_pred))
print("Recall:", recall_score(y_test_sabra, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test_sabra, y_pred))

# Fine-tune model with RandomizedSearchCV
parameters = {
    'classifier__min_samples_split': range(10, 300, 20),
    'classifier__max_depth': range(1, 30, 2),
    'classifier__min_samples_leaf': range(1, 15, 3)
}

search = RandomizedSearchCV(
    estimator=pipeline_sabra,
    param_distributions=parameters,
    scoring='accuracy',
    cv=5,
    n_iter=7,
    verbose=3,
    refit=True,
    random_state=28
)
search.fit(X_train_sabra, y_train_sabra)

print("Best parameters:", search.best_params_)
print("Best cross-validation accuracy:", search.best_score_)

# Evaluate fine-tuned model
best_model = search.best_estimator_
y_pred_tuned = best_model.predict(X_test_sabra)
print("Tuned Accuracy:", accuracy_score(y_test_sabra, y_pred_tuned))
print("Tuned Precision:", precision_score(y_test_sabra, y_pred_tuned))
print("Tuned Recall:", recall_score(y_test_sabra, y_pred_tuned))

# Save model and pipeline
joblib.dump(tree_model, 'decision_tree_model_sabra.pkl')
joblib.dump(pipeline_sabra, 'full_pipeline_sabra.pkl')