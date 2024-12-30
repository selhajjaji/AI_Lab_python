# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 14:33:18 2024

@author: m
"""

# Alex- Data Collection, Loading & Exploration

from ucimlrepo import fetch_ucirepo 
import pandas as pd
# fetch dataset
youtube_spam_collection = fetch_ucirepo(id=380)  
# data (as pandas dataframes)
X = youtube_spam_collection.data.features 
y = youtube_spam_collection.data.targets
# variable informationpr
print(youtube_spam_collection.variables)
# Convert to pandas dataframe
# The dataset has 5 features and 1 target variable
df = pd.DataFrame(X)
df['CLASS'] = y
# Basic data exploration
print(df.head())
print(df.describe())
# Select only two columns
# CONTENT: The content of the comment
# CLASS: The class of the comment (1 = spam, 0 = ham)
df = df[['CONTENT', 'CLASS']] 

# Sabra- Data Pre-Processing & Transformation

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import nltk
import string

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('punkt_tab')
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
