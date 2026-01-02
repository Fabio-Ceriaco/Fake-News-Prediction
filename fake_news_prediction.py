# Work Flow
# 1. Import Necessary Libraries
# 2. Data Collection
# 3. Data Preprocessing
# 4. Stemming the Text Data
# 5 . Separate Data into Features and Labels
# 6. Converting of Text Data To Numerical Data
# 7. Split the Data into Training and Testing Sets
# 8. Train a Machine Learning Model
# 9. Evaluate the Model
# 10. Make Predictions

# =================================================================================#
#                       1. Import Necessary Libraries                             #
# =================================================================================#

import pandas as pd
import numpy as np
import re

# Make sure to download stopwords
# stopwords is a collection of commonly used words in a language that are often ignored in text analysis.
# Examples of stopwords in English include "the", "is", "in", "and", etc.
from nltk.corpus import stopwords

# Make sure to download PorterStemmer
# PorterStemmer is an algorithm for reducing words to their root or base form.
# For example, "running", "runner", and "ran" would all be reduced to "run".
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# accuracy_score - A function to calculate the accuracy of the model's predictions.
# classification_report - A function that provides a detailed report of the model's performance, including precision, recall, and F1-score.
# confusion_matrix - A function that creates a matrix to visualize the performance of the model by showing the true vs predicted classifications.
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Download NLTK resources
import nltk

nltk.download("stopwords")  # Download stopwords


# Print English stopwords
# print(stopwords.words("english"))

# =================================================================================#
#                                2. Data Collection                               #
# =================================================================================#

# Data Structure
# Id: Unique identifier for each news article.
# Title: The title of the news article.
# Author: The author of the news article.
# Text: The main content of the news article.
# Label: A binary label indicating whether the news article is 0 real news or 1 fake news


# Pandas options
pd.set_option("display.max_columns", 6)


# Load the Dataset

data = pd.read_csv("news_data.csv")


# =================================================================================#
#                            3. Data Preprocessing                                #
# =================================================================================#

# Analyze the Dataset

print("\nCheck the shape of the training dataset:")
print(data.shape)

print("\nFirst 5 rows of the training dataset:")
print(data.head())

print("\nCheck for missing values in the training dataset:")
print(data.isnull().sum())

# In this dataset we have some missing values in the 'text', 'author' and 'title' columns.
# Has we have a large dataset we can drop the missing values or fill the null values with empty strings.
# If we have a small dataset we can use techniques like imputation to fill in the missing values.
# Imputation is the process of replacing missing data with substituted values.

# Replace null values with empty strings in both train_data

data = data.fillna("")


# Merging the 'title', 'author', and 'text' columns into a single 'content' column
# in both datasets train_data and test_data
data = data.fillna("")


# Merging the 'title', 'author', and 'text' columns into a single 'content' column
data["content"] = data[["author", "title", "text"]].astype(str).agg(" ".join, axis=1)


print("\nColumns in the dataset after merging:")
print(data.head())


# ==================================================================================#
#                           4. Stemming the text data                                  #
# ==================================================================================#


# Stemming is the process of reducing words to their root or base form.
# Examples of stemming include reducing "running", "runner", and "ran" to "run".
# This helps in normalizing the text data and reducing the dimensionality of the feature space.

port_stem = PorterStemmer()
STOP_WORDS = set(
    stopwords.words("english")
)  # we have converted to set for faster lookup if we don't use set it will be a list by default


def stemming(content: str):
    # Remove all non-alphabetic characters
    stemming_content = re.sub("[^a-zA-Z]", " ", content)
    # Convert to lowercase
    stemming_content = stemming_content.lower()
    # Split into words
    stemming_content = stemming_content.split()
    # Apply stemming to each word and remove stopwords
    stemming_content = [
        port_stem.stem(word) for word in stemming_content if not word in STOP_WORDS
    ]
    # Join the words back into a single string
    stemming_content = " ".join(stemming_content)
    return stemming_content


# Apply stemming to the 'content' column in both training and testing sets
# After applying stemming the text data will be in its root form which will help in better feature extraction and model performance.

data["content"] = data["content"].astype(str).apply(stemming)


print("\nStemming applied to training data:")
print(data["content"])

# ==================================================================================#
#                  5. Separate Data into Features and Labels                       #
# ==================================================================================#

X = data["content"].values  # Features (text data)
y = data["label"].values  # Labels (real or fake)


print("\nFeatures and Labels separated:")
print("Features (X):", X)
print("Labels (y):", y)  # Labels (real or fake) 1 for fake news and 0 for real news


# ==================================================================================#
#               6. Converting of Text Data To Numerical Data                        #
# ==================================================================================#

vectorizer = TfidfVectorizer()

# TF-IDF (Term Frequency-Inverse Document Frequency) is a numerical representation of text data that reflects
# the importance of words in a document relative to a collection of documents (corpus),
# he counts the number of times a word appears in a document and adjusts it based on how common the word is across all documents.
# It helps in converting text data into a format that can be used by machine learning algorithms.
# The TF-IDF vectorizer transforms the text data into a matrix of TF-IDF features,
# that we can use for training machine learning models because model don't understand raw text data.


vectorizer.fit(X)  # Learn vocabulary and idf from training set

X = vectorizer.transform(X)  # Transform text data to feature vectors

print("\nText data converted to numerical data using TF-IDF:")
print(X)  # X is now a sparse matrix representation of the text data

# ==================================================================================#
#            7. Split the Data into Training and Testing Sets                       #
# ==================================================================================#

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=2
)


# =================================================================================#
#                     8. Train a Machine Learning Model                           #
# =================================================================================#


# ====================Logistic Regression Math Explanation ========================#


# Logistic Regression is a simple and effective algorithm for binary classification.
# This model uses sigmoid curve predict the probability of a data point belonging to a particular class (fake or real news).
# A sigmoid function is a mathematical function that maps any input value to a value between 0 and 1,
# and represents the probability of the input belonging to the positive class.
# Y = 1 / (1 + e^(-z)) where z is a linear combination of input features and model coefficients.
# In the sigmoid function, as the input value (z) increases, the output value (Y) approaches 1,
# indicating a higher probability of belonging to the positive class. Conversely, as the input value decreases, the output value approaches 0,
# indicating a lower probability of belonging to the positive class.
# In the sigmoid function z is calculated as z = w.X + b
# where w is the weight vector, X is the input feature vector, and b is the bias term.
# X - input features (TF-IDF vectors of news articles) and represents the characteristics of the news articles
# w - weights (coefficients learned by the model during training) and represents the importance of each feature
# b - bias (intercept term learned by the model during training) and represents the baseline prediction when all features are zero
# z - linear combination of input features and model coefficients and represents the weighted sum of the input features plus the bias term
# Y - predicted probability of the news article being fake (1) or real (0) news.


model = LogisticRegression()

# Train the model on the training data

model.fit(X_train, y_train)

# =================================================================================#
#                        9. Evaluate the Model                                     #
# =================================================================================#

# Accuracy Score

# Make prediction on training data

X_train_prediction = model.predict(X_train)

# Calculate accuracy on training data
training_data_accuracy = accuracy_score(X_train_prediction, y_train)

print("\nAccuracy on Training data:", training_data_accuracy)

# classification Report
training_classification_report = classification_report(y_train, X_train_prediction)

print("\nClassification Report on Training data:")
print(training_classification_report)

# confusion Matrix
training_confusion_matrix = confusion_matrix(y_train, X_train_prediction)
print("\nConfusion Matrix on Training data:")
print(training_confusion_matrix)


# Make prediction on test data

X_test_prediction = model.predict(X_test)

# Calculate accuracy on test data

test_data_accuracy = accuracy_score(X_test_prediction, y_test)

print("Accuracy on Test data:", test_data_accuracy)

# classification Report
test_classification_report = classification_report(y_test, X_test_prediction)

print("\nClassification Report on Test data:")
print(test_classification_report)

# confusion Matrix

test_confusion_matrix = confusion_matrix(y_test, X_test_prediction)

print("\nConfusion Matrix on Test data:")
print(test_confusion_matrix)


# ==================================================================================#
#                         10. Make Predictions                                     #
# ==================================================================================#

input_news = X_test[1]  # Example news article from the test set

prediction = model.predict(input_news)


if prediction[0] == 0:
    print("The news article is Real.")
else:
    print("The news article is Fake.")
