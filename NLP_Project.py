# -*- coding: utf-8 -*-
"""
@author: COMP237-402-24F - Group 2
"""

# Alex- Data Collection, Loading & Exploration

from ucimlrepo import fetch_ucirepo 
import pandas as pd
# fetch dataset
youtube_spam_collection = fetch_ucirepo(id=380)  
# data (as pandas dataframes)
X = youtube_spam_collection.data.features 
y = youtube_spam_collection.data.targets
# variable information
print(youtube_spam_collection.variables)
# Convert to pandas dataframe
# The dataset has 5 features and 1 target variable
df = pd.DataFrame(X)
df['CLASS'] = y
# Basic data exploration
print(df.head(5)) # Prints first 5 rows
print(df.sample(5)) # Prints 5 random rows
print(df.describe())
# Select only two columns
# CONTENT: The content of the comment
# CLASS: The class of the comment (1 = spam, 0 = not-spam)
df = df[['CONTENT', 'CLASS']] 

# Sabra- Data Pre-Processing & Transformation

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import nltk
import string

# Download necessary NLTK data files
nltk.download('stopwords')
from nltk.corpus import stopwords

# Define stopwords
stop_words = set(stopwords.words('english'))

# Define a function to preprocess the text
def preprocess_text(text):
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    # Convert to lowercase and remove punctuation
    tokens = [word.lower() for word in tokens if word not in string.punctuation]
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Apply preprocessing to the 'CONTENT' column
df['CONTENT'] = df['CONTENT'].apply(preprocess_text)

# Transform the text data using Bag of Words (CountVectorizer)
vectorizer = CountVectorizer()
X_count = vectorizer.fit_transform(df['CONTENT'])

# Display feature names and the shape of the transformed data
print("\nBag of Words feature names (first 10):")
print(vectorizer.get_feature_names_out()[:10])  # First 10 features
print("\nShape of transformed data (Bag of Words):", X_count.shape)

# Apply TF-IDF transformation
tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_count)

# Display the shape of the TF-IDF transformed data
print("\nShape of transformed data (TF-IDF):", X_tfidf.shape)

# Daria - data shuffling, splitting, training a Naive Bayes classifier, and performing cross-validation

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
import numpy as np

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split the dataset into training (75%) and testing (25%) sets
# Separate features (X) and labels (y)
train_size = int(0.75 * len(df))
train_data = df[:train_size]
test_data = df[train_size:]

X_train = vectorizer.transform(train_data['CONTENT'])
y_train = train_data['CLASS']
X_test = vectorizer.transform(test_data['CONTENT'])
y_test = test_data['CLASS']

# Fit the training data into a Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Evaluate the classifier on the testing set
accuracy = nb_classifier.score(X_test, y_test)
print("\nAccuracy on the testing set:", accuracy)

# Perform 5-fold cross-validation on the training data
cv_scores = cross_val_score(nb_classifier, X_train, y_train, cv=5)

# Print the mean accuracy from cross-validation
print("\nMean accuracy from 5-fold cross-validation:", np.mean(cv_scores))

# Karim - Model Testing and New Comment Classification
from sklearn.metrics import confusion_matrix, classification_report
y_pred = nb_classifier.predict(X_test) # Predict on the test set
conf_matrix = confusion_matrix(y_test, y_pred) # Generate confusion matrix
print("Confusion Matrix:\n", conf_matrix) # Print confusion matrix
class_report = classification_report(y_test, y_pred, target_names=["Not-Spam", "Spam"]) # Generate classification report
print("\nClassification Report:\n", class_report) # Print classification report

# Test the model with new comments
new_comments = [
    "Win a free iPhone now! Click the link below.",
    "Subscribe to our channel for more awesome videos!",
    "Earn $1000 a day by working from home.",
    "Thank you for sharing such an informative video.",
    "Exclusive offer! Buy one, get one free.",
    "Great content, really enjoyed this!",
    "This is a great video!",
    "I love this content!",
    "Check out my channel!",
    "Subscribe to my channel!",
    "Nice tutorial, very helpful.",
    "Click here to win a free iPhone!"
]

# Preprocess the new comments
new_comments_preprocessed = [
    " ".join([word.lower() for word in comment.split() if word not in string.punctuation])
    for comment in new_comments
]

# Transform new comments to TF-IDF representation
new_comments_vectorized = tfidf_transformer.transform(vectorizer.transform(new_comments_preprocessed))

# Predict classes for the new comments
new_predictions = nb_classifier.predict(new_comments_vectorized)

# Display predictions for the new comments
for i, comment in enumerate(new_comments):
    print(f"Comment: {comment}\nPredicted Class: {'Spam' if new_predictions[i] == 1 else 'Not-Spam'}\n")