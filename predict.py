import pickle
import pandas as pd
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import pickle

# load data
# dataset = loadtxt('test.csv', delimiter=",")
# split data into X and y
# X_test = dataset[:,0:8]
# y_test = dataset[:,8]
dataset = pd.read_csv("/content/train.csv")

x_tes = dataset['text']
y_test = dataset['label']

# Define the TF-IDF vectorizer for unigrams
tfidf_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 1))

# Transform the text data to TF-IDF vectors
x_test = tfidf_vectorizer.fit_transform(x_tes)
# load model from file
loaded_model = pickle.load(open("/content/pima_model.pkl", "rb"))

# make predictions for test data
y_pred = loaded_model.predict(x_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
#Confusion Matrix
conf_matrix = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(conf_matrix)

# Classification Report
class_report = classification_report(y_test, predictions)
print("Classification Report:")
print(class_report)
