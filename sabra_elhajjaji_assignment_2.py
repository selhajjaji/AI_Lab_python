# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 16:36:19 2025

@author: m
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# Load MNIST dataset
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', parser='auto')
list(mnist.keys())
print(mnist.DESCR)
MINST_sabra = pd.DataFrame(data=mnist.data, columns=mnist.feature_names)
MINST_sabra['target'] = mnist.target

# Assign data and target
X_sabra = mnist.data.to_numpy()
y_sabra = mnist.target.to_numpy()

# Print types and shapes
print(f"Type of X_sabra: {type(X_sabra)}")
print(f"Type of y_sabra: {type(y_sabra)}")
print(f"Shape of X_sabra: {X_sabra.shape}")
print(f"Shape of y_sabra: {y_sabra.shape}")

# Create variables based on name
some_digit12 = X_sabra[3].reshape(28, 28)
some_digit13 = X_sabra[8].reshape(28, 28)
some_digit14 = X_sabra[1].reshape(28, 28)

# Plot the digits
plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.imshow(some_digit12, cmap='gray')
plt.title("Digit 3")
plt.subplot(1, 3, 2)
plt.imshow(some_digit13, cmap='gray')
plt.title("Digit 8")
plt.subplot(1, 3, 3)
plt.imshow(some_digit14, cmap='gray')
plt.title("Digit 1")
plt.show()

# Preprocess the data
y_sabra = y_sabra.astype(np.uint8)
y_transformed = np.where((y_sabra >= 0) & (y_sabra <= 1), 0,
                         np.where((y_sabra >= 2) & (y_sabra <= 3), 1,
                                  np.where((y_sabra >= 4) & (y_sabra <= 5), 2,
                                           np.where((y_sabra >= 6) & (y_sabra <= 7), 3, 4))))

# Print frequencies of each class
class_counts = np.bincount(y_transformed)
print("Class Frequencies:", class_counts)
plt.bar(range(5), class_counts)
plt.title("Class Frequencies")
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.show()

# Split data into train and test
X_train, X_test = X_sabra[:55000], X_sabra[55000:]
y_train, y_test = y_transformed[:55000], y_transformed[55000:]

# Naive Bayes Classifier
print("Training Naive Bayes Classifier...")
NB_clf_sabra = MultinomialNB()
NB_clf_sabra.fit(X_train, y_train)

# Cross-validation
nb_cv_scores = cross_val_score(NB_clf_sabra, X_train, y_train, cv=3)
print("Naive Bayes Cross-Validation Scores:", nb_cv_scores)

# Test accuracy
nb_test_accuracy = NB_clf_sabra.score(X_test, y_test)
print("Naive Bayes Test Accuracy:", nb_test_accuracy)

# Confusion matrix
y_pred_nb = NB_clf_sabra.predict(X_test)
conf_matrix_nb = confusion_matrix(y_test, y_pred_nb)
print("Naive Bayes Confusion Matrix:\n", conf_matrix_nb)

# Predict on selected digits
pred_digit12_nb = NB_clf_sabra.predict([X_sabra[3]])
pred_digit13_nb = NB_clf_sabra.predict([X_sabra[8]])
pred_digit14_nb = NB_clf_sabra.predict([X_sabra[1]])
print("Naive Bayes Predictions:")
print(f"Digit 3: {pred_digit12_nb[0]}, Digit 8: {pred_digit13_nb[0]}, Digit 1: {pred_digit14_nb[0]}")

# Logistic Regression Classifier
print("Training Logistic Regression Classifier...")
LR_clf_sabra = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=900, tol=0.1)
LR_clf_sabra.fit(X_train, y_train)

# Cross-validation
lr_cv_scores = cross_val_score(LR_clf_sabra, X_train, y_train, cv=3)
print("Logistic Regression Cross-Validation Scores:", lr_cv_scores)

# Test accuracy
lr_test_accuracy = LR_clf_sabra.score(X_test, y_test)
print("Logistic Regression Test Accuracy:", lr_test_accuracy)

# Confusion matrix
y_pred_lr = LR_clf_sabra.predict(X_test)
conf_matrix_lr = confusion_matrix(y_test, y_pred_lr)
print("Logistic Regression Confusion Matrix:\n", conf_matrix_lr)

# Predict on selected digits
pred_digit12_lr = LR_clf_sabra.predict([X_sabra[3]])
pred_digit13_lr = LR_clf_sabra.predict([X_sabra[8]])
pred_digit14_lr = LR_clf_sabra.predict([X_sabra[1]])
print("Logistic Regression Predictions:")
print(f"Digit 3: {pred_digit12_lr[0]}, Digit 8: {pred_digit13_lr[0]}, Digit 1: {pred_digit14_lr[0]}")

# Precision and Recall
precision = precision_score(y_test, y_pred_lr, average='weighted')
recall = recall_score(y_test, y_pred_lr, average='weighted')
print("Logistic Regression Precision:", precision)
print("Logistic Regression Recall:", recall)