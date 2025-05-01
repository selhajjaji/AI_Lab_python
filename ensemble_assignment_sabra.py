# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 12:34:44 2025

@author: m
"""
# ensemble_assignment_sabra.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score

# Load the dataset
df_sabra = pd.read_csv('D:/Centennial College-Winter2025/Supervised Learning/ensemble_learning/pima-indians-diabetes.csv', header=None)
df_sabra.columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]

# Initial investigation
print(df_sabra.info())
print(df_sabra.isnull().sum())
print(df_sabra.describe())
print(df_sabra["Outcome"].value_counts())

# Preprocessing
transformer_sabra = StandardScaler()
X = df_sabra.drop("Outcome", axis=1)
y = df_sabra["Outcome"]

# Splitting
X_train_sabra, X_test_sabra, y_train_sabra, y_test_sabra = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Scaling
X_train_sabra = transformer_sabra.fit_transform(X_train_sabra)
X_test_sabra = transformer_sabra.transform(X_test_sabra)

# Classifiers
clf1_E = LogisticRegression(max_iter=1400)
clf2_E = RandomForestClassifier()
clf3_E = SVC(probability=True)
clf4_E = DecisionTreeClassifier(criterion="entropy", max_depth=42)
clf5_E = ExtraTreesClassifier()

# ---------------------- Exercise #1: Hard Voting ---------------------- #
hard_voting_clf = VotingClassifier(
    estimators=[
        ('lr_E', clf1_E), 
        ('rf_E', clf2_E), 
        ('svc_E', clf3_E), 
        ('dt_E', clf4_E), 
        ('et_E', clf5_E)
    ],
    voting='hard'
)

hard_voting_clf.fit(X_train_sabra, y_train_sabra)
hard_preds = hard_voting_clf.predict(X_test_sabra[:3])
print("\nHard Voting Predictions (first 3 instances):")
print("Actual:", y_test_sabra.iloc[:3].values)
print("Voting:", hard_preds)

# Individual classifiers
print("\nPredictions from individual classifiers:")
for name, clf in [('lr_E', clf1_E), ('rf_E', clf2_E), ('svc_E', clf3_E), ('dt_E', clf4_E), ('et_E', clf5_E)]:
    clf.fit(X_train_sabra, y_train_sabra)
    preds = clf.predict(X_test_sabra[:3])
    print(f"{name}: {preds}")

# ---------------------- Exercise #2: Soft Voting ---------------------- #
soft_voting_clf = VotingClassifier(
    estimators=[
        ('lr_E', clf1_E), 
        ('rf_E', clf2_E), 
        ('svc_E', clf3_E), 
        ('dt_E', clf4_E), 
        ('et_E', clf5_E)
    ],
    voting='soft'
)

soft_voting_clf.fit(X_train_sabra, y_train_sabra)
soft_preds = soft_voting_clf.predict(X_test_sabra[:3])
print("\nSoft Voting Predictions (first 3 instances):")
print("Actual:", y_test_sabra.iloc[:3].values)
print("Voting:", soft_preds)

# ---------------------- Exercise #3: Pipelines and Evaluation ---------------------- #
pipeline1_sabra = Pipeline([
    ('scaler', StandardScaler()),
    ('et_E', clf5_E)
])

pipeline2_sabra = Pipeline([
    ('scaler', StandardScaler()),
    ('dt_E', clf4_E)
])

# Cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores1 = cross_val_score(pipeline1_sabra, X, y, cv=cv)
scores2 = cross_val_score(pipeline2_sabra, X, y, cv=cv)

print("\n10-Fold CV Mean Accuracy:")
print("Extra Trees Pipeline:", scores1.mean())
print("Decision Tree Pipeline:", scores2.mean())

# Confusion matrices and metrics
for pipe, name in [(pipeline1_sabra, "Extra Trees"), (pipeline2_sabra, "Decision Tree")]:
    pipe.fit(X_train_sabra, y_train_sabra)
    y_pred = pipe.predict(X_test_sabra)
    print(f"\n{name} Pipeline - Evaluation:")
    print("Confusion Matrix:\n", confusion_matrix(y_test_sabra, y_pred))
    print("Precision:", precision_score(y_test_sabra, y_pred))
    print("Recall:", recall_score(y_test_sabra, y_pred))
    print("Accuracy:", accuracy_score(y_test_sabra, y_pred))

# ---------------------- Exercise #4: Randomized Grid Search ---------------------- #
param_grid_28 = {
    'et_E__n_estimators': np.arange(10, 3001, 20),
    'et_E__max_depth': np.arange(1, 1001, 2)
}

grid_search_28 = RandomizedSearchCV(
    estimator=pipeline1_sabra,
    param_distributions=param_grid_28,
    n_iter=20,
    scoring='accuracy',
    cv=cv,
    random_state=42,
    n_jobs=-1
)

grid_search_28.fit(X_train_sabra, y_train_sabra)

print("\nBest Parameters from Grid Search:")
print(grid_search_28.best_params_)
print("Best CV Accuracy:", grid_search_28.best_score_)

# Final evaluation
best_model = grid_search_28.best_estimator_
final_preds = best_model.predict(X_test_sabra)

print("\nFinal Tuned Model - Evaluation:")
print("Confusion Matrix:\n", confusion_matrix(y_test_sabra, final_preds))
print("Precision:", precision_score(y_test_sabra, final_preds))
print("Recall:", recall_score(y_test_sabra, final_preds))
print("Accuracy:", accuracy_score(y_test_sabra, final_preds))

